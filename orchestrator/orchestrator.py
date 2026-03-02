"""
ProbeAnalyzer: Main interface for linear probe analysis.

This is the primary entry point for the library. It handles:
- Model loading via the appropriate backend
- Activation extraction
- Parallel probe training
- Result aggregation
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union

import numpy as np

from easyprobe.extractors.base import ActivationExtractor, SEQ_LENGTHS_KEY
from easyprobe.extractors.transformerlens import TransformerLensExtractor
from easyprobe.extractors.nnsight import NNSightExtractor
from easyprobe.models.data_models import (
    BackendOption,
    ComponentOption,
    ComponentSpec,
    DeviceOption,
    FeatureTimingReport,
    LayerSpec,
    ModelTimingReport,
    MultiFeatureSharedPromptsData,
    MultiFeatureSeparatePromptsData,
    PositionOption,
    PositionSpec,
    ProbeData,
    ProbeTask,
    ProbeTimingReport,
    ProbeTrainingRecord,
    ProbeType,
    SingleFeatureData,
    AggregationMethod
)
from easyprobe.models.probe_results import ProbeResults, MultiFeatureProbeResults
from easyprobe.probing.normalize import ActivationNormalizer
from easyprobe.probing.train import train_single_probe
from easyprobe.models.linear_probe import LinearProbe
from easyprobe.util.validation import validate_layer_spec
from easyprobe.util.helpers import parse_position_spec, normalize_component_spec
from easyprobe.util.profiler import ProbeProfiler
from datetime import datetime

logger = logging.getLogger(__name__)


class ProbeOrchestrator:
    """
    Main interface for linear probe analysis.

    ProbeOrchestrator handles the full pipeline:
    1. Load model (via TransformerLens)
    2. Extract activations for your prompts
    3. Train linear probes in parallel
    4. Return results with visualizations

    Example:
        orchestrator = ProbeOrchestrator("pythia-410m")
        results = orchestrator.probe(
            prompts=["I love this!", "I hate this."],
            labels=[1, 0],
        )
        results.show()

    For quick experiments, use the module-level quick_probe() function:
        from easyprobe import quick_probe
        results = quick_probe("pythia-410m", prompts, labels)
    """

    def __init__(
        self,
        model: str,
        backend: BackendOption = BackendOption.AUTO,
        device: DeviceOption = DeviceOption.AUTO,
        revision: Optional[str] = None,
        remote: bool = False,
        revisions: Optional[Union[list[str], dict[str, str], list[tuple[str, str]]]] = None,
    ):
        """
        Initialize the probe analyzer.

        Args:
            model: Model identifier
                - TransformerLens: "pythia-410m", "gpt2-small", etc.
                - NNSight: Any HuggingFace model ID (e.g., "allenai/OLMo-2-7B-1124")
            backend: Which library to use for activation extraction (BackendOption enum)
                - BackendOption.AUTO or BackendOption.TRANSFORMERLENS: Use TransformerLens
                - BackendOption.NNSIGHT: Use NNSight (for models not supported by TransformerLens)
            device: Device to run on (DeviceOption enum) for activation extraction
                - DeviceOption.AUTO: Use CUDA if available, else MPS, else CPU
            revision: Git revision (branch, tag, or commit) for HuggingFace models.
                      Only used with NNSight backend. E.g., "stage1-step896000" for OLMo-3.
            remote: Unused (kept for backward compatibility)
            revisions: Provide multiple revisions for multi-stage/multi-model comparison mode.
        """
        self.model_name = model
        self.backend_name = backend
        self.device = device
        self.revision = revision
        self.remote = remote
        self.profiler = ProbeProfiler(verbose=True)

        self.is_multi_model = revisions is not None
        if self.is_multi_model:
            self.revisions: dict[str, str] = {}
            if isinstance(revisions, dict):
                self.revisions = revisions
            elif isinstance(revisions, list):
                for item in revisions:
                    if isinstance(item, tuple) and len(item) == 2:
                        self.revisions[item[1]] = item[0]
                    else:
                        self.revisions[str(item)] = str(item)
            
            # Defer loading models until .probe() is called to save memory
            self.extractor = None
            self.model_config = None
            self.model_loading_s = 0.0
        else:
            # Load model immediately for single-model mode
            self.extractor, self.model_config = self._load_model(model, backend, device, revision, remote)
            self.model_loading_s = self.profiler.get_timing("model_loading")

    def _load_model(
        self,
        model: str,
        backend: BackendOption,
        device: DeviceOption,
        revision: Optional[str],
        remote: bool,
    ) -> tuple[ActivationExtractor, dict]:
        """Load the model and return extractor + config."""
        revision_str = f" (revision: {revision})" if revision else ""
        self.profiler.log(f"Loading model: {model}{revision_str}...")

        with self.profiler.time("model_loading"):
            extractor = self._create_extractor(model, backend, device, revision, remote)
            config = extractor.get_model_config()

        self.profiler.log(
            f"Model loaded in {self.profiler.get_timing('model_loading'):.2f}s "
            f"({config['n_layers']} layers, hidden_dim={config['hidden_dim']}, device={config['device']})"
        )
        return extractor, config

    def _create_extractor(
        self,
        model: str,
        backend: BackendOption,
        device: DeviceOption,
        revision: Optional[str],
        remote: bool,
    ) -> ActivationExtractor:
        """Create the appropriate activation extractor for the given backend."""

        if backend in (BackendOption.AUTO, BackendOption.TRANSFORMERLENS):
            return TransformerLensExtractor(model, device)

        if backend == BackendOption.NNSIGHT:
            return NNSightExtractor(model, device, revision=revision)

        raise ValueError(f"Unknown backend: {backend}")

    def _extract_and_normalize_activations(
        self,
        prompts: list[str],
        layers: LayerSpec,
        components: ComponentSpec,
        position: PositionSpec,
        batch_size: int,
        activation_checkpoint_path: Optional[str] = None,
        auto_cleanup: bool = True,
    ) -> tuple[dict, ActivationNormalizer, float, float]:
        """
        Extract and z-score normalize activations (shared by single and multi-feature modes).

        Returns:
            Tuple of:
            - Dictionary mapping (layer, component) -> normalized activations
            - ActivationNormalizer with fitted params (for storing in LinearProbe)
            - extraction_s: time spent extracting activations
            - normalization_s: time spent normalizing activations
        """
        # Parse and normalize specifications
        layer_list = validate_layer_spec(layers, self.model_config["n_layers"])

        if components is None:
            components = [ComponentOption.RESID]
        components = normalize_component_spec(components)

        # Extract activations
        n_prompts = len(prompts)
        component_names = [c.value for c in components]
        self.profiler.log(f"Extracting activations for {n_prompts} prompts across {len(layer_list)} layers, components={component_names}...")

        with self.profiler.time("extraction"):
            activations = self.extractor.extract_activations(
                prompts=prompts,
                layers=layer_list,
                components=components,
                position=position,
                batch_size=batch_size,
                activation_checkpoint_path=activation_checkpoint_path,
                auto_cleanup=auto_cleanup,
            )

        extraction_s = self.profiler.get_timing("extraction")
        self.profiler.log(f"Activation extraction completed in {extraction_s:.2f}s")

        # Z-score normalize activations (skip SEQ_LENGTHS_KEY which is metadata, not activations)
        with self.profiler.time("normalization"):
            normalizer = ActivationNormalizer()
            normalized_activations = {
                key: normalizer.fit_transform(acts, key)
                for key, acts in activations.items()
                if key != SEQ_LENGTHS_KEY
            }
            # Preserve sequence lengths metadata if present
            if SEQ_LENGTHS_KEY in activations:
                normalized_activations[SEQ_LENGTHS_KEY] = activations[SEQ_LENGTHS_KEY]

        normalization_s = self.profiler.get_timing("normalization")
        self.profiler.log(f"Normalization completed in {normalization_s:.2f}s")

        return normalized_activations, normalizer, extraction_s, normalization_s

    def _create_probe_tasks(
        self,
        normalized_activations: dict,
        labels: np.ndarray,
        position: PositionSpec,
        regularization: float,
        probe_type: ProbeType,
        include_selectivity: bool,
        random_trials: int,
        normalizer: Optional[ActivationNormalizer] = None,
    ) -> list[ProbeTask]:
        """
        Create probe tasks from normalized activations (shared logic).

        When using PositionOption.ALL, filters out samples where the position
        index exceeds the actual sequence length (i.e., padded positions).

        Returns:
            List of ProbeTask objects ready for training
        """
        tasks = []

        # Extract sequence lengths if available (for PositionOption.ALL)
        seq_lengths = normalized_activations.pop(SEQ_LENGTHS_KEY, None)

        # Check if any activations remain after removing metadata
        if len(normalized_activations) == 0:
            raise ValueError(
                "No activations found. This may indicate that extraction failed or "
                "all activations were filtered out during normalization."
            )

        # Parse position specification
        first_key = list(normalized_activations.keys())[0]
        seq_len = normalized_activations[first_key].shape[1] if normalized_activations[first_key].ndim == 3 else 1
        task_specs = parse_position_spec(position, seq_len)

        for (layer, component), acts in normalized_activations.items():
            for indices_to_use in task_specs:
                # Determine what to store in the ProbeTask.position field
                if position in [PositionOption.LAST, PositionOption.MEAN]:
                    task_position = position
                else:
                    task_position = indices_to_use

                # If acts has sequence dim, select/average the tokens
                if acts.ndim == 3:
                    if len(indices_to_use) == 1 and position != PositionOption.MEAN:
                        idx = indices_to_use[0]
                        try:
                            current_acts = acts[:, idx, :]
                        except IndexError:
                            continue

                        # For PositionOption.ALL, filter out samples where this position is padding
                        current_labels = labels
                        if seq_lengths is not None and position == PositionOption.ALL:
                            # Only include samples where seq_lengths > idx
                            valid_mask = seq_lengths > idx
                            if not np.any(valid_mask):
                                # No valid samples for this position, skip
                                continue
                            current_acts = current_acts[valid_mask]
                            current_labels = labels[valid_mask]
                    else:
                        current_acts = acts[:, indices_to_use, :].mean(axis=1)
                        current_labels = labels
                else:
                    current_acts = acts
                    current_labels = labels

                # Skip positions with insufficient samples per class for train/test split
                # Need at least 5 samples per class for meaningful stratified split
                unique, counts = np.unique(current_labels, return_counts=True)
                if len(unique) < 2 or np.min(counts) < 5:
                    continue

                # Get normalization params for this key
                norm_mean, norm_std = None, None
                key = (layer, component)
                if normalizer is not None:
                    try:
                        norm_mean, norm_std = normalizer.get_params(key)
                    except KeyError:
                        pass

                tasks.append(
                    ProbeTask(
                        layer=layer,
                        component=component,
                        position=task_position,
                        activations=current_acts,
                        labels=current_labels,
                        regularization=regularization,
                        probe_type=probe_type,
                        include_selectivity=include_selectivity,
                        random_trials=random_trials,
                        norm_mean=norm_mean,
                        norm_std=norm_std,
                    )
                )

        return tasks

    def _train_probes(
        self,
        tasks: list[ProbeTask],
        max_workers: Optional[int],
        feature_name: Optional[str] = None,
    ) -> tuple[list, float]:
        """
        Train probes in parallel (shared logic).

        Args:
            tasks: List of ProbeTask objects
            max_workers: Number of parallel workers (None = all CPUs)
            feature_name: Optional feature name for logging context

        Returns:
            Tuple of:
            - List of LinearProbe objects
            - training_s: wall-clock time for training
        """
        n_tasks = len(tasks)
        feature_prefix = f" [{feature_name}]" if feature_name else ""
        timing_key = f"training_{feature_name}" if feature_name else "training"

        self.profiler.log(f"{feature_prefix} Training {n_tasks} probes...")

        with self.profiler.time(timing_key):
            results = []

            if max_workers == 1:
                # Sequential training with progress
                for i, task in enumerate(tasks, 1):
                    result = train_single_probe(task)
                    results.append(result)
                    pos_str = result.position.value if hasattr(result.position, 'value') else result.position
                    training_time_str = f" ({result.training_time:.2f}s)" if result.training_time else ""
                    self.profiler.log(f"{feature_prefix} Probe {i}/{n_tasks}: layer={result.layer}, component={result.component.value}, pos={pos_str} -> acc={result.accuracy:.1%}{training_time_str}")
            else:
                # Parallel training with progress using as_completed
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_idx = {executor.submit(train_single_probe, task): i for i, task in enumerate(tasks)}

                    # Collect results as they complete
                    completed = 0
                    results_dict = {}
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        result = future.result()
                        results_dict[idx] = result
                        completed += 1
                        pos_str = result.position.value if hasattr(result.position, 'value') else result.position
                        training_time_str = f" ({result.training_time:.2f}s)" if result.training_time else ""
                        self.profiler.log(f"{feature_prefix} Probe {completed}/{n_tasks}: layer={result.layer}, component={result.component.value}, pos={pos_str} -> acc={result.accuracy:.1%}{training_time_str}")

                    # Restore original order
                    results = [results_dict[i] for i in range(n_tasks)]

        training_s = self.profiler.get_timing(timing_key)

        # Find best result
        best_result = max(results, key=lambda r: r.accuracy)
        pos_str = best_result.position.value if hasattr(best_result.position, 'value') else best_result.position
        self.profiler.log(f"{feature_prefix} Training completed in {training_s:.2f}s. Best: layer={best_result.layer}, component={best_result.component.value}, pos={pos_str}, acc={best_result.accuracy:.1%}")

        return results, training_s

    def _build_feature_timing_report(
        self,
        feature_name: str,
        n_samples: int,
        extraction_s: float,
        normalization_s: float,
        training_s: float,
        results: list,
    ) -> FeatureTimingReport:
        """Build a FeatureTimingReport from probe results."""
        # Build per-probe records
        probe_records = []
        for r in results:
            probe_records.append(
                ProbeTrainingRecord(
                    layer=r.layer,
                    component=r.component,
                    position=r.position if isinstance(r.position, PositionOption) else r.position,
                    pid=None,  # LinearProbe doesn't track process ID
                    start_time=None,  # LinearProbe doesn't track start/end times
                    end_time=None,
                    duration_s=r.training_time if r.training_time else 0.0,
                    accuracy=r.accuracy,
                    accuracy_std=r.accuracy_std,
                    random_baseline=r.random_baseline,
                    selectivity=r.selectivity,
                    n_samples=r.n_samples,
                    weights=r.weights,
                    bias=r.bias,
                )
            )

        # Find best result
        best_result = max(results, key=lambda r: r.accuracy)

        return FeatureTimingReport(
            feature_name=feature_name,
            n_samples=n_samples,
            n_probes=len(results),
            extraction_s=extraction_s,
            normalization_s=normalization_s,
            training_s=training_s,
            probe_records=probe_records,
            best_layer=best_result.layer,
            best_component=best_result.component,
            best_position=best_result.position,
            best_accuracy=best_result.accuracy,
            best_selectivity=best_result.selectivity,
            best_weights=best_result.weights,
            best_bias=best_result.bias,
        )

    def probe(
        self,
        data: ProbeData,
        layers: LayerSpec = "all",
        components: ComponentSpec = None,
        position: PositionSpec = PositionOption.LAST,
        # Probe settings
        regularization: float = 1.0,
        probe_type: ProbeType = ProbeType.CLASSIFICATION,
        # Validation settings
        include_selectivity: bool = True,
        random_trials: int = 5,
        # Infrastructure settings
        batch_size: int = 8,
        max_workers: Optional[int] = None,
        activation_checkpoint_path: Optional[str] = None,
        auto_cleanup: bool = True,
    ) -> Union[ProbeResults, MultiFeatureProbeResults, dict]:
        """
        Train linear probes on model activations.

        Args:
            data: ProbeData object specifying prompts and labels.
                Three types supported:
                - SingleFeatureData(prompts, labels): Basic one-to-one
                - MultiFeatureSharedPromptsData(prompts, features): Multiple features, same prompts (most efficient)
                - MultiFeatureSeparatePromptsData(features): Multiple features, different prompts

            layers: Which layers to probe
                - "all": All layers (default)
                - [0, 5, 10]: Specific layers
                - range(0, 16): Range of layers
            position: Which token position to extract
                - PositionOption.LAST: Last token (default, best for autoregressive)
                - int: Specific position index (0-indexed)
                - list[int]: Multiple specific indices ([0, 5, 10])
            components: Which components to probe (list of ComponentOption enums or strings)
                - None: Residual stream only (default)
                - [ComponentOption.RESID]: Residual stream only
                - [ComponentOption.RESID, ComponentOption.ATTN, ComponentOption.MLP]: All components
                - ["resid", "attn", "mlp"]: All components (strings also accepted)

            regularization: L2 regularization strength
                - Higher = simpler probe, less overfitting
                - Default 1.0 works well for most cases
            probe_type: Type of probe
                - "classification": Predict categories (accuracy metric)

            include_selectivity: Whether to compute random baseline
                - True: Train on shuffled labels to verify signal
                - Selectivity = accuracy - random_baseline
            random_trials: Number of random shuffles for selectivity

            batch_size: Batch size for activation extraction
            max_workers: Maximum parallel workers for probe training
                - None: Use all available CPUs
                - 1: Sequential (useful for debugging)

        Returns:
            - ProbeResults: For single feature data
            - MultiFeatureProbeResults: For multi-feature data
            - dict: For multi-model iterations (if revisions passed to __init__)

        Both provide visualization methods, data export, and summary statistics.


        Examples:
            # Single feature
            data = SingleFeatureData(prompts=["I love this!", "I hate this!"], labels=[1, 0])
            results = analyzer.probe(data)
            results.show()

            # Multiple features (shared prompts - most efficient!)
            data = MultiFeatureSharedPromptsData(
                prompts=prompts,
                features={
                    "sentiment": sentiment_labels,
                    "topic": topic_labels,
                    "formality": formality_labels,
                }
            )
            results = analyzer.probe(data)
            results.plot_feature_comparison()
            results["sentiment"].plot_layer_accuracy()

            # Multiple features (separate prompts)
            data = MultiFeatureSeparatePromptsData(
                features={
                    "sentiment": (sentiment_prompts, sentiment_labels),
                    "deception": (deception_prompts, deception_labels),
                }
            )
            results = analyzer.probe(data)
            for feature_name, feature_results in results.items():
                logger.info(f"{feature_name}: best layer = {feature_results.best_layer}")
        """
        if self.is_multi_model:
            return self._probe_multi_model(
                data=data,
                layers=layers,
                position=position,
                components=components,
                regularization=regularization,
                probe_type=probe_type,
                include_selectivity=include_selectivity,
                random_trials=random_trials,
                batch_size=batch_size,
                max_workers=max_workers,
                activation_checkpoint_path=activation_checkpoint_path,
                auto_cleanup=auto_cleanup,
            )
        # Determine probe mode based on data type
        if isinstance(data, SingleFeatureData):
            return self._probe_single_feature(
                data, layers, position, components, regularization,
                probe_type, include_selectivity,
                random_trials, batch_size, max_workers,
                activation_checkpoint_path=activation_checkpoint_path, auto_cleanup=auto_cleanup
            )
        elif isinstance(data, MultiFeatureSharedPromptsData):
            return self._probe_multi_feature_shared(
                data, layers, position, components, regularization,
                probe_type, include_selectivity,
                random_trials, batch_size, max_workers,
                activation_checkpoint_path=activation_checkpoint_path, auto_cleanup=auto_cleanup
            )
        elif isinstance(data, MultiFeatureSeparatePromptsData):
            return self._probe_multi_feature_separate(
                data, layers, position, components, regularization,
                probe_type, include_selectivity,
                random_trials, batch_size, max_workers,
                activation_checkpoint_path=activation_checkpoint_path, auto_cleanup=auto_cleanup
            )
        else:
            raise TypeError(f"Invalid data type: {type(data)}. Expected ProbeData.")

    def _probe_multi_model(
        self,
        data: ProbeData,
        layers: LayerSpec,
        position: PositionSpec,
        components: ComponentSpec,
        regularization: float,
        probe_type: ProbeType,
        include_selectivity: bool,
        random_trials: int,
        batch_size: int,
        max_workers: Optional[int],
        activation_checkpoint_path: Optional[str] = None,
        auto_cleanup: bool = True,
    ) -> dict:
        """Probe multiple revisions of the same model structure seamlessly."""
        results_dict = {}
        checkpoint_paths = []
        
        self.profiler.log(f"Multi-model mode: {len(self.revisions)} models.")
        for stage_name, revision in self.revisions.items():
            self.profiler.log(f"\n=== Model Stage: {stage_name} ({revision}) ===")
            
            # Create a single-model orchestrator
            sub_orchestrator = ProbeOrchestrator(
                model=self.model_name,
                backend=self.backend_name,
                device=self.device,
                revision=revision,
                remote=self.remote,
            )
            
            # Unique checkpoint path
            current_checkpoint_path = None
            if activation_checkpoint_path:
                checkpoint_name = revision.replace("/", "_").replace("-", "_")
                current_checkpoint_path = f"{activation_checkpoint_path}_{checkpoint_name}"
                checkpoint_paths.append(current_checkpoint_path)
            
            # Since we manage cleanup here, auto_cleanup is passed as False securely
            sub_results = sub_orchestrator.probe(
                data=data,
                layers=layers,
                position=position,
                components=components,
                regularization=regularization,
                probe_type=probe_type,
                include_selectivity=include_selectivity,
                random_trials=random_trials,
                batch_size=batch_size,
                max_workers=max_workers,
                activation_checkpoint_path=current_checkpoint_path,
                auto_cleanup=False,  
            )
            results_dict[stage_name] = sub_results
            
            # Immediately free the single orchestrator (and its massive activations dict) to save memory
            del sub_orchestrator
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Handle post-loop cleanup
        if auto_cleanup and checkpoint_paths:
            from easyprobe.util.helpers import cleanup_activation_checkpoints
            cleaned = cleanup_activation_checkpoints(checkpoint_paths)
            if cleaned:
                self.profiler.log(f"Cleaned up multi-model checkpoint directories.")
                
        return results_dict

    def _probe_single_feature(
        self,
        data: SingleFeatureData,
        layers: LayerSpec,
        position: PositionSpec,
        components: ComponentSpec,
        regularization: float,
        probe_type: ProbeType,
        include_selectivity: bool,
        random_trials: int,
        batch_size: int,
        max_workers: Optional[int],
        activation_checkpoint_path: Optional[str] = None,
        auto_cleanup: bool = True,
    ) -> ProbeResults:
        """Probe a single feature."""
        run_start_time = time.perf_counter()

        prompts = data.prompts
        labels = data.labels
        labels_array = np.array(labels)

        # Extract and normalize activations
        normalized_activations, normalizer, extraction_s, normalization_s = self._extract_and_normalize_activations(
            prompts, layers, components, position, batch_size,
            activation_checkpoint_path=activation_checkpoint_path, auto_cleanup=auto_cleanup
        )

        # Create probe tasks (pass normalizer so norm params are stored in each probe)
        tasks = self._create_probe_tasks(
            normalized_activations, labels_array, position,
            regularization, probe_type,
            include_selectivity, random_trials,
            normalizer=normalizer
        )

        # Train probes
        results, training_s = self._train_probes(tasks, max_workers, feature_name=None)

        # Build timing report
        total_s = time.perf_counter() - run_start_time
        logger.info(f"Total time: {total_s:.2f}s (model loading: {self.model_loading_s:.2f}s, extraction: {extraction_s:.2f}s, training: {training_s:.2f}s)")
        feature_report = self._build_feature_timing_report(
            feature_name="default",
            n_samples=len(prompts),
            extraction_s=extraction_s,
            normalization_s=normalization_s,
            training_s=training_s,
            results=results,
        )
        model_report = ModelTimingReport(
            model_name=self.model_name,
            model_loading_s=self.model_loading_s,
            extraction_s=extraction_s,
            feature_reports=[feature_report],
        )
        timing_report = ProbeTimingReport(
            run_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_s=total_s,
            model_reports=[model_report],
        )

        return ProbeResults(
            results=results,
            prompts=prompts,
            labels=labels_array,
            model_name=self.model_name,
            timing_report=timing_report,
        )

    def _probe_multi_feature_shared(
        self,
        data: MultiFeatureSharedPromptsData,
        layers: LayerSpec,
        position: PositionSpec,
        components: ComponentSpec,
        regularization: float,
        probe_type: ProbeType,
        include_selectivity: bool,
        random_trials: int,
        batch_size: int,
        max_workers: Optional[int],
        activation_checkpoint_path: Optional[str] = None,
        auto_cleanup: bool = True,
    ) -> MultiFeatureProbeResults:
        """
        Probe multiple features with shared prompts.

        This is the most efficient mode: activations are extracted once and reused
        for all features.
        """
        run_start_time = time.perf_counter()
        prompts = data.prompts

        # Extract and normalize activations ONCE for all features
        normalized_activations, normalizer, extraction_s, normalization_s = self._extract_and_normalize_activations(
            prompts, layers, components, position, batch_size,
            activation_checkpoint_path=activation_checkpoint_path, auto_cleanup=auto_cleanup
        )

        # Train probes for each feature
        feature_results = {}
        feature_timing_reports = []
        total_training_s = 0.0

        logger.info(f"Multi-feature mode: {data.num_features} features ({', '.join(data.feature_names)})")

        for i, (feature_name, feature_labels) in enumerate(data.features.items(), 1):
            logger.info(f"Feature {i}/{data.num_features}: {feature_name}")
            labels_array = np.array(feature_labels)

            # Create probe tasks (pass normalizer so norm params are stored in each probe)
            tasks = self._create_probe_tasks(
                normalized_activations, labels_array, position,
                regularization, probe_type,
                include_selectivity, random_trials,
                normalizer=normalizer
            )

            # Train probes
            results, training_s = self._train_probes(tasks, max_workers, feature_name=feature_name)
            total_training_s += training_s

            # Build feature timing report (extraction_s=0 since shared)
            feature_report = self._build_feature_timing_report(
                feature_name=feature_name,
                n_samples=len(prompts),
                extraction_s=0.0,  # Shared extraction, not counted per-feature
                normalization_s=normalization_s / data.num_features,  # Split evenly
                training_s=training_s,
                results=results,
            )
            feature_timing_reports.append(feature_report)

            feature_results[feature_name] = ProbeResults(
                results=results,
                prompts=prompts,
                labels=labels_array,
                model_name=self.model_name,
            )

        # Build timing report
        total_s = time.perf_counter() - run_start_time
        logger.info(f"Total time: {total_s:.2f}s (model loading: {self.model_loading_s:.2f}s, extraction: {extraction_s:.2f}s, training: {total_training_s:.2f}s)")
        model_report = ModelTimingReport(
            model_name=self.model_name,
            model_loading_s=self.model_loading_s,
            extraction_s=extraction_s,
            feature_reports=feature_timing_reports,
        )
        timing_report = ProbeTimingReport(
            run_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_s=total_s,
            model_reports=[model_report],
        )

        return MultiFeatureProbeResults(
            feature_results=feature_results,
            model_name=self.model_name,
            timing_report=timing_report,
        )

    def _probe_multi_feature_separate(
        self,
        data: MultiFeatureSeparatePromptsData,
        layers: LayerSpec,
        position: PositionSpec,
        components: ComponentSpec,
        regularization: float,
        probe_type: ProbeType,
        include_selectivity: bool,
        random_trials: int,
        batch_size: int,
        max_workers: Optional[int],
        activation_checkpoint_path: Optional[str] = None,
        auto_cleanup: bool = True,
    ) -> MultiFeatureProbeResults:
        """
        Probe multiple features with separate prompts.

        Activations are extracted separately for each feature.
        """
        run_start_time = time.perf_counter()
        feature_results = {}
        feature_timing_reports = []
        total_extraction_s = 0.0
        total_training_s = 0.0

        logger.info(f"Multi-feature mode (separate prompts): {data.num_features} features ({', '.join(data.feature_names)})")

        for i, (feature_name, (prompts, labels)) in enumerate(data.features.items(), 1):
            logger.info(f"Feature {i}/{data.num_features}: {feature_name} ({len(prompts)} samples)")
            labels_array = np.array(labels)

            # Create a feature-specific checkpoint dir if checkpointing is enabled
            feature_activation_checkpoint_path = f"{activation_checkpoint_path}_{feature_name}" if activation_checkpoint_path else None

            # Extract and normalize activations for this feature
            normalized_activations, normalizer, extraction_s, normalization_s = self._extract_and_normalize_activations(
                prompts, layers, components, position, batch_size,
                activation_checkpoint_path=feature_activation_checkpoint_path, auto_cleanup=auto_cleanup
            )
            total_extraction_s += extraction_s

            # Create probe tasks (pass normalizer so norm params are stored in each probe)
            tasks = self._create_probe_tasks(
                normalized_activations, labels_array, position,
                regularization, probe_type,
                include_selectivity, random_trials,
                normalizer=normalizer
            )

            # Train probes
            results, training_s = self._train_probes(tasks, max_workers, feature_name=feature_name)
            total_training_s += training_s

            # Build feature timing report
            feature_report = self._build_feature_timing_report(
                feature_name=feature_name,
                n_samples=len(prompts),
                extraction_s=extraction_s,
                normalization_s=normalization_s,
                training_s=training_s,
                results=results,
            )
            feature_timing_reports.append(feature_report)

            feature_results[feature_name] = ProbeResults(
                results=results,
                prompts=prompts,
                labels=labels_array,
                model_name=self.model_name,
            )

        # Build timing report
        total_s = time.perf_counter() - run_start_time
        logger.info(f"Total time: {total_s:.2f}s (model loading: {self.model_loading_s:.2f}s, extraction: {total_extraction_s:.2f}s, training: {total_training_s:.2f}s)")
        model_report = ModelTimingReport(
            model_name=self.model_name,
            model_loading_s=self.model_loading_s,
            extraction_s=total_extraction_s,
            feature_reports=feature_timing_reports,
        )
        timing_report = ProbeTimingReport(
            run_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_s=total_s,
            model_reports=[model_report],
        )

        return MultiFeatureProbeResults(
            feature_results=feature_results,
            model_name=self.model_name,
            timing_report=timing_report,
        )

    def predict(
        self,
        text: str,
        probe: LinearProbe,
        aggregation: AggregationMethod = AggregationMethod.LAST,
        threshold: float = 0.5,
    ) -> tuple[float, int]:
        """
        Run inference with a trained probe on a single text.

        This method:
        1. Extracts activations for the text (at the probe's layer/component)
        2. Applies the probe
        3. Aggregates the result (if multiple tokens)

        Args:
            text: Input text
            probe: Trained LinearProbe object
            aggregation: How to aggregate scores across tokens (MAX, MEAN, LAST)
            threshold: Decision threshold

        Returns:
            Tuple of (score, label)
        """
        # 1. Extract activations (force batch_size=1)
        # We need ALL tokens for aggregation flexibility, so use PositionOption.ALL
        activations_dict = self.extractor.extract_activations(
            prompts=[text],
            layers=[probe.layer],
            components=[probe.component],
            position=PositionOption.ALL, 
            batch_size=1
        )
        
        # Get specific tensor
        key = (probe.layer, probe.component)
        # Shape: [1, seq_len, hidden_dim]
        activation = activations_dict[key]
        
        # Remove batch dim -> [seq_len, hidden_dim]
        if activation.ndim == 3:
            activation = activation[0]
            
        # 2. Predict using probe's helper
        return probe.predict_on_sequence(activation, aggregation, threshold)

    def __repr__(self) -> str:
        return (
            f"ProbeOrchestrator(model='{self.model_name}', "
            f"backend='{self.backend_name}', device='{self.device}')"
        )

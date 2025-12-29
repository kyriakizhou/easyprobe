"""
ProbeAnalyzer: Main interface for linear probe analysis.

This is the primary entry point for the library. It handles:
- Model loading via the appropriate backend
- Activation extraction
- Parallel probe training
- Result aggregation
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union

import numpy as np

from easyprobe.extractors.base import ActivationExtractor
from easyprobe.extractors.transformerlens import TransformerLensExtractor
from easyprobe.extractors.nnsight import NNSightExtractor
from easyprobe.datamodels import (
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
)
from easyprobe.probe_results import ProbeResults, MultiFeatureProbeResults
from easyprobe.probing.normalize import ActivationNormalizer, NormalizationMethod
from easyprobe.probing.train import train_single_probe
from easyprobe.util.validation import validate_layer_spec
from easyprobe.util.helpers import parse_position_spec, normalize_component_spec
from easyprobe.util.profiler import ProbeProfiler
from datetime import datetime


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
                - DeviceOption.CPU, DeviceOption.CUDA, DeviceOption.MPS: Use specific device
            revision: Git revision (branch, tag, or commit) for HuggingFace models.
                      Only used with NNSight backend. E.g., "stage1-step896000" for OLMo-3.
            remote: Unused (kept for backward compatibility)
        """
        self.model_name = model
        self.backend_name = backend
        self.device = device
        self.revision = revision
        self.remote = remote
        self.profiler = ProbeProfiler(verbose=True)

        # Load model
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
        self.profiler.log(f"[EasyProbe] Loading model: {model}{revision_str}...")

        with self.profiler.time("model_loading"):
            extractor = self._create_extractor(model, backend, device, revision, remote)
            config = extractor.get_model_config()

        self.profiler.log(
            f"[EasyProbe] Model loaded in {self.profiler.get_timing('model_loading'):.2f}s "
            f"({config['n_layers']} layers, hidden_dim={config['hidden_dim']})"
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
        normalize: NormalizationMethod,
    ) -> tuple[dict, float, float]:
        """
        Extract and normalize activations (shared by single and multi-feature modes).

        Returns:
            Tuple of:
            - Dictionary mapping (layer, component, head) -> normalized activations
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
        self.profiler.log(f"[EasyProbe] Extracting activations for {n_prompts} prompts across {len(layer_list)} layers, components={component_names}...")

        with self.profiler.time("extraction"):
            activations = self.extractor.extract_activations(
                prompts=prompts,
                layers=layer_list,
                components=components,
                position=position,
                batch_size=batch_size,
            )

        extraction_s = self.profiler.get_timing("extraction")
        self.profiler.log(f"[EasyProbe] Activation extraction completed in {extraction_s:.2f}s")

        # Normalize activations
        with self.profiler.time("normalization"):
            normalizer = ActivationNormalizer(normalize)
            normalized_activations = {
                key: normalizer.fit_transform(acts, key)
                for key, acts in activations.items()
            }

        normalization_s = self.profiler.get_timing("normalization")
        self.profiler.log(f"[EasyProbe] Normalization completed in {normalization_s:.2f}s")

        return normalized_activations, extraction_s, normalization_s

    def _create_probe_tasks(
        self,
        normalized_activations: dict,
        labels: np.ndarray,
        position: PositionSpec,
        regularization: float,
        probe_type: ProbeType,
        include_selectivity: bool,
        random_trials: int,
    ) -> list[ProbeTask]:
        """
        Create probe tasks from normalized activations (shared logic).

        Returns:
            List of ProbeTask objects ready for training
        """
        tasks = []

        # Parse position specification
        first_key = list(normalized_activations.keys())[0]
        seq_len = normalized_activations[first_key].shape[1] if normalized_activations[first_key].ndim == 3 else 1
        task_specs = parse_position_spec(position, seq_len)

        for (layer, component, _), acts in normalized_activations.items():
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
                    else:
                        current_acts = acts[:, indices_to_use, :].mean(axis=1)
                else:
                    current_acts = acts

                tasks.append(
                    ProbeTask(
                        layer=layer,
                        component=component,
                        position=task_position,
                        activations=current_acts,
                        labels=labels,
                        regularization=regularization,
                        probe_type=probe_type,
                        include_selectivity=include_selectivity,
                        random_trials=random_trials,
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
            - List of ProbeResult objects
            - training_s: wall-clock time for training
        """
        n_tasks = len(tasks)
        feature_prefix = f" [{feature_name}]" if feature_name else ""
        timing_key = f"training_{feature_name}" if feature_name else "training"

        self.profiler.log(f"[EasyProbe]{feature_prefix} Training {n_tasks} probes...")

        with self.profiler.time(timing_key):
            results = []

            if max_workers == 1:
                # Sequential training with progress
                for i, task in enumerate(tasks, 1):
                    result = train_single_probe(task)
                    results.append(result)
                    pos_str = result.position.value if hasattr(result.position, 'value') else result.position
                    self.profiler.log(f"[EasyProbe]{feature_prefix} Probe {i}/{n_tasks}: layer={result.layer}, component={result.component.value}, pos={pos_str} -> acc={result.accuracy:.1%} ({result.training_duration_s:.2f}s)")
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
                        self.profiler.log(f"[EasyProbe]{feature_prefix} Probe {completed}/{n_tasks}: layer={result.layer}, component={result.component.value}, pos={pos_str} -> acc={result.accuracy:.1%} ({result.training_duration_s:.2f}s)")

                    # Restore original order
                    results = [results_dict[i] for i in range(n_tasks)]

        training_s = self.profiler.get_timing(timing_key)

        # Find best result
        best_result = max(results, key=lambda r: r.accuracy)
        pos_str = best_result.position.value if hasattr(best_result.position, 'value') else best_result.position
        self.profiler.log(f"[EasyProbe]{feature_prefix} Training completed in {training_s:.2f}s. Best: layer={best_result.layer}, component={best_result.component.value}, pos={pos_str}, acc={best_result.accuracy:.1%}")

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
                    pid=r.pid,
                    start_time=r.start_time,
                    end_time=r.end_time,
                    duration_s=r.training_duration_s,
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
        # What to probe
        layers: LayerSpec = "all",
        position: PositionSpec = PositionOption.LAST,
        components: ComponentSpec = None,
        # Probe settings
        regularization: float = 1.0,
        normalize: NormalizationMethod = NormalizationMethod.ZSCORE,
        probe_type: ProbeType = ProbeType.CLASSIFICATION,
        # Validation settings
        include_selectivity: bool = True,
        random_trials: int = 5,
        # Infrastructure settings
        batch_size: int = 8,
        max_workers: Optional[int] = None,
    ) -> Union[ProbeResults, MultiFeatureProbeResults]:
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
            normalize: How to normalize activations
                - "zscore": Recommended, makes layers comparable
                - "minmax": Scale to [0, 1]
                - "none": No normalization
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
                print(f"{feature_name}: best layer = {feature_results.best_layer}")
        """
        # Determine probe mode based on data type
        if isinstance(data, SingleFeatureData):
            return self._probe_single_feature(
                data, layers, position, components, regularization,
                normalize, probe_type, include_selectivity,
                random_trials, batch_size, max_workers
            )
        elif isinstance(data, MultiFeatureSharedPromptsData):
            return self._probe_multi_feature_shared(
                data, layers, position, components, regularization,
                normalize, probe_type, include_selectivity,
                random_trials, batch_size, max_workers
            )
        elif isinstance(data, MultiFeatureSeparatePromptsData):
            return self._probe_multi_feature_separate(
                data, layers, position, components, regularization,
                normalize, probe_type, include_selectivity,
                random_trials, batch_size, max_workers
            )
        else:
            raise TypeError(f"Invalid data type: {type(data)}. Expected ProbeData.")

    def _probe_single_feature(
        self,
        data: SingleFeatureData,
        layers: LayerSpec,
        position: PositionSpec,
        components: ComponentSpec,
        regularization: float,
        normalize: NormalizationMethod,
        probe_type: ProbeType,
        include_selectivity: bool,
        random_trials: int,
        batch_size: int,
        max_workers: Optional[int],
        run_start_time: Optional[float] = None,
    ) -> ProbeResults:
        """Probe a single feature."""
        if run_start_time is None:
            run_start_time = time.perf_counter()

        prompts = data.prompts
        labels = data.labels
        labels_array = np.array(labels)

        # Extract and normalize activations
        normalized_activations, extraction_s, normalization_s = self._extract_and_normalize_activations(
            prompts, layers, components, position, batch_size, normalize
        )

        # Create probe tasks
        tasks = self._create_probe_tasks(
            normalized_activations, labels_array, position,
            regularization, probe_type,
            include_selectivity, random_trials
        )

        # Train probes
        results, training_s = self._train_probes(tasks, max_workers, feature_name=None)

        # Build timing report
        total_s = time.perf_counter() - run_start_time
        print(f"[EasyProbe] Total time: {total_s:.2f}s (model loading: {self.model_loading_s:.2f}s, extraction: {extraction_s:.2f}s, training: {training_s:.2f}s)")
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
        normalize: NormalizationMethod,
        probe_type: ProbeType,
        include_selectivity: bool,
        random_trials: int,
        batch_size: int,
        max_workers: Optional[int],
    ) -> MultiFeatureProbeResults:
        """
        Probe multiple features with shared prompts.

        This is the most efficient mode: activations are extracted once and reused
        for all features.
        """
        run_start_time = time.perf_counter()
        prompts = data.prompts

        # Extract and normalize activations ONCE for all features
        normalized_activations, extraction_s, normalization_s = self._extract_and_normalize_activations(
            prompts, layers, components, position, batch_size, normalize
        )

        # Train probes for each feature
        feature_results = {}
        feature_timing_reports = []
        total_training_s = 0.0

        print(f"[EasyProbe] Multi-feature mode: {data.num_features} features ({', '.join(data.feature_names)})")

        for i, (feature_name, feature_labels) in enumerate(data.features.items(), 1):
            print(f"\n[EasyProbe] === Feature {i}/{data.num_features}: {feature_name} ===")
            labels_array = np.array(feature_labels)

            # Create probe tasks
            tasks = self._create_probe_tasks(
                normalized_activations, labels_array, position,
                regularization, probe_type,
                include_selectivity, random_trials
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
        print(f"\n[EasyProbe] Total time: {total_s:.2f}s (model loading: {self.model_loading_s:.2f}s, extraction: {extraction_s:.2f}s, training: {total_training_s:.2f}s)")
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
        normalize: NormalizationMethod,
        probe_type: ProbeType,
        include_selectivity: bool,
        random_trials: int,
        batch_size: int,
        max_workers: Optional[int],
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

        print(f"[EasyProbe] Multi-feature mode (separate prompts): {data.num_features} features ({', '.join(data.feature_names)})")

        for i, (feature_name, (prompts, labels)) in enumerate(data.features.items(), 1):
            print(f"\n[EasyProbe] === Feature {i}/{data.num_features}: {feature_name} ({len(prompts)} samples) ===")
            labels_array = np.array(labels)

            # Extract and normalize activations for this feature
            normalized_activations, extraction_s, normalization_s = self._extract_and_normalize_activations(
                prompts, layers, components, position, batch_size, normalize
            )
            total_extraction_s += extraction_s

            # Create probe tasks
            tasks = self._create_probe_tasks(
                normalized_activations, labels_array, position,
                regularization, probe_type,
                include_selectivity, random_trials
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
        print(f"\n[EasyProbe] Total time: {total_s:.2f}s (model loading: {self.model_loading_s:.2f}s, extraction: {total_extraction_s:.2f}s, training: {total_training_s:.2f}s)")
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

    def __repr__(self) -> str:
        return (
            f"ProbeOrchestrator(model='{self.model_name}', "
            f"backend='{self.backend_name}', device='{self.device}')"
        )

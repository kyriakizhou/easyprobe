"""
easyprobe: One-liner linear probes for mechanistic interpretability.

Stop writing boilerplate. Start finding features.

Example:
    from easyprobe import ProbeOrchestrator

    orchestrator = ProbeOrchestrator("pythia-410m")
    results = orchestrator.probe(
        prompts=["I love this!", "I hate this."],
        labels=[1, 0],
    )
    results.show()


"""

from easyprobe.orchestrate import ProbeOrchestrator
from easyprobe.probe_results import ProbeResults, MultiFeatureProbeResults
from easyprobe.datamodels import (
    BackendOption,
    ComponentOption,
    ComponentSpec,
    DeviceOption,
    LayerSpec,
    MultiFeatureSharedPromptsData,
    MultiFeatureSeparatePromptsData,
    PositionOption,
    PositionSpec,
    ProbeData,
    ProbeResult,
    ProbeTask,
    ProbeType,
    SingleFeatureData,
)
from easyprobe.probing.normalize import NormalizationMethod
from easyprobe.extractors.base import ActivationExtractor
from easyprobe.extractors.transformerlens import TransformerLensExtractor
from easyprobe.extractors.nnsight import NNSightExtractor


def quick_probe(
    model: str,
    data: ProbeData,
    backend: BackendOption = BackendOption.AUTO,
    device: DeviceOption = DeviceOption.AUTO,
    **kwargs,
):
    """
    One-liner probe function for quick experiments.

    Args:
        model: Model name (e.g., "pythia-410m", "gpt2-small")
        data: ProbeData object (SingleFeatureData, MultiFeatureSharedPromptsData, or MultiFeatureSeparatePromptsData)
        backend: Backend to use
        device: Device to run on
        **kwargs: Additional arguments passed to analyzer.probe()

    Returns:
        ProbeResults or MultiFeatureProbeResults

    Example:
        # Single feature
        data = SingleFeatureData(
            prompts=["I love this!", "I hate this!"],
            labels=[1, 0]
        )
        results = quick_probe(model="pythia-410m", data=data)
        results.show()

        # Multiple features
        data = MultiFeatureSharedPromptsData(
            prompts=prompts,
            features={"sentiment": labels1, "topic": labels2}
        )
        results = quick_probe(model="pythia-410m", data=data)
        results.plot_feature_comparison()
    """
    orchestrator = ProbeOrchestrator(model, backend=backend, device=device)
    return orchestrator.probe(data, **kwargs)


__version__ = "0.1.0"

__all__ = [
    # Main classes
    "ProbeOrchestrator",
    "ProbeResults",
    "MultiFeatureProbeResults",
    "ProbeResult",
    # Convenience functions
    "quick_probe",
    # Data classes
    "SingleFeatureData",
    "MultiFeatureSharedPromptsData",
    "MultiFeatureSeparatePromptsData",
    "ProbeData",
    # Enums
    "ProbeType",
    "NormalizationMethod",
    "PositionOption",
    "ComponentOption",
    "DeviceOption",
    "BackendOption",
    # Type aliases
    "LayerSpec",
    "PositionSpec",
    "ComponentSpec",
    # Backends (for advanced users)
    "ActivationExtractor",
    "TransformerLensExtractor",
    "NNSightExtractor",
]

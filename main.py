"""
EasyProbe: Comprehensive demonstration of all use scenarios.

This script demonstrates 6 key scenarios for linear probing with easyprobe:
1. Single feature (factuality), last token, all layers (basic layer sweep)
2. Single feature (factuality), last token, all components (component comparison)
3. Single feature (factuality), all positions, all layers (position analysis)
4. Multiple features (factuality + topic, shared prompts), last token, all layers
5. Multiple features (factuality + topic, separate prompts), last token, all layers
6. OLMo-3 training stages comparison for factuality detection

Datasets used:
- factuality_large: 800 prompts for true/false factuality (binary: true=1, false=0)
- topics_large: 1000 prompts for topic classification (binary: math=1, climate=0)
- factuality_topic_shared: 800 prompts with both factuality and topic labels (multi-label)
"""

import gc
import os
import shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from easyprobe import ProbeOrchestrator, SingleFeatureData, MultiFeatureSharedPromptsData, MultiFeatureSeparatePromptsData
from easyprobe.datamodels import ComponentOption, PositionOption, BackendOption
from easyprobe.visualization import plot_multi_model_heatmap, generate_multi_model_report
# Import factuality datasets
from easyprobe.data.factuality import (
    fact_prompts_large,
    fact_labels_large,
    topics_prompts_large,
    topics_labels_large,
    scenario4_prompts,
    scenario4_factuality_labels,
    scenario4_topic_labels,
)

TEST_CORRECTNESS = True
USE_NNSIGHT = False  # Set to True to use NNSight backend, False for TransformerLens

# Backend configuration
# Production uses OLMo-3 7B final stage (requires A100 40GB+)
PRODUCTION_MODEL = "allenai/Olmo-3-1025-7B"  # Same as OLMO3_BASE_MODEL below
PRODUCTION_REVISION = "stage3-step9000"  # Final stage (Long Context)

if USE_NNSIGHT:
    # Use gpt2 (124M) for fast local testing, OLMo-3 7B for production
    llm = "gpt2" if TEST_CORRECTNESS else PRODUCTION_MODEL
    llm_revision = None if TEST_CORRECTNESS else PRODUCTION_REVISION
    backend = BackendOption.NNSIGHT
else:
    llm = "pythia-70m" if TEST_CORRECTNESS else PRODUCTION_MODEL
    llm_revision = None if TEST_CORRECTNESS else PRODUCTION_REVISION
    backend = BackendOption.TRANSFORMERLENS

# Model comparison settings
max_workers = 11
batch_size = 4 if TEST_CORRECTNESS else 1  # batch_size=1 for OLMo-3 7B on A100 (long conversations use lots of memory)

# OLMo-3 training stages for scenario 6 (uses NNSight backend)
# These are the 3 training stages with their final checkpoints:
# - Stage 1: Initial pretraining (5.93T tokens on Dolma 3)
# - Stage 2: Mid-training (100B tokens on Dolmino-mix)
# - Stage 3: Long context training (50B tokens on Longmino-mix)
OLMO3_BASE_MODEL = "allenai/Olmo-3-1025-7B"
# Final checkpoint of each training stage:
# - Stage 1: 5.93T tokens pretraining on dolma3_6T-mix-1025
# - Stage 2: 100B tokens mid-training on dolma3-dolmino-mix-1025
# - Stage 3: 50B tokens long context on dolma3-longmino-mix-1025
OLMO3_STAGES = [
    ("stage1-step1413814", "Stage 1 (Pretraining)"),
    ("stage2-step47684", "Stage 2 (Mid-training)"),
    ("stage3-step11921", "Stage 3 (Long Context)"),
]


def clear_gpu_memory():
    """Clear GPU memory between scenarios."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def scenario_1_basic_layer_sweep():
    """
    Scenario 1: Probe Single Feature on Last Token Position of All Layers

    This is the most basic and common use case:
    - One feature (factuality: true vs false statements)
    - Last token position (standard for autoregressive models)
    - All layers (find where the feature emerges)

    Use this to answer: "At which layer does the model encode factuality?"
    """
    print("\n" + "="*80)
    print("SCENARIO 1: Basic Layer Sweep (Factuality Detection, Last Token, All Layers)")
    print("="*80)

    orchestrator = ProbeOrchestrator(llm, backend=backend, revision=llm_revision)

    # Prepare data - detect true statements (true=1, false=0)
    data = SingleFeatureData(
        prompts=fact_prompts_large,
        labels=fact_labels_large
    )
    print(f"Dataset: {len(fact_prompts_large)} prompts ({sum(fact_labels_large)} true, {len(fact_labels_large) - sum(fact_labels_large)} false)")

    results = orchestrator.probe(
        data=data,
        layers="all",
        position=PositionOption.LAST,
        components=None,
        include_selectivity=True,
        random_trials=2,
        max_workers=max_workers,
        batch_size=batch_size,
        checkpoint_dir="checkpoints/scenario1",
    )

    print(f"\nBest layer: {results.best_layer}")
    print(f"Best accuracy: {results.best_result.accuracy:.1%}")
    print(f"Selectivity: {results.best_result.selectivity:.1%}")

    results.plot_heatmap_interactive(output_path="scenario1.html", show=False)
    results.generate_report(output_path="scenario1_report.html", show=False)
    print("✓ Saved: scenario1.html")

    return results


def scenario_2_component_comparison():
    """
    Scenario 2: Probe Single Feature on Last Token Position of All Components

    Compare how factuality is encoded in:
    - Residual stream (cumulative representation)
    - Attention outputs (what the model is "looking at")
    - MLP outputs (non-linear transformations)

    Use this to answer: "Which component encodes factuality better?"
    """
    print("\n" + "="*80)
    print("SCENARIO 2: Component Comparison (Attention vs MLP vs Residual)")
    print("="*80)

    orchestrator = ProbeOrchestrator(llm, backend=backend, revision=llm_revision)

    data = SingleFeatureData(
        prompts=fact_prompts_large,
        labels=fact_labels_large
    )
    print(f"Dataset: {len(fact_prompts_large)} prompts ({sum(fact_labels_large)} true, {len(fact_labels_large) - sum(fact_labels_large)} false)")

    results = orchestrator.probe(
        data=data,
        layers="all",
        position=PositionOption.LAST,
        components=[ComponentOption.RESID, ComponentOption.ATTN, ComponentOption.MLP],
        include_selectivity=True,
        random_trials=2,
        max_workers=max_workers,
        batch_size=batch_size,
        checkpoint_dir="checkpoints/scenario2",
    )

    print("\nComponent comparison:")
    for component in [ComponentOption.RESID, ComponentOption.ATTN, ComponentOption.MLP]:
        component_results = [r for r in results.results if r.component == component]
        if component_results:
            best = max(component_results, key=lambda r: r.accuracy)
            print(f"  {component.value:6s}: Layer {best.layer:2d}, Accuracy {best.accuracy:.1%}, Selectivity {best.selectivity:.1%}")

    results.plot_heatmap_interactive(output_path="scenario2.html", show=False)
    results.generate_report(output_path="scenario2_report.html", show=False)
    print("✓ Saved: scenario2.html")

    return results


def scenario_3_position_analysis():
    """
    Scenario 3: Probe Single Feature on All Token Positions of All Layers

    Use this to answer: "Does factuality emerge at specific token positions?"
    """
    print("\n" + "="*80)
    print("SCENARIO 3: Position Analysis (All Tokens, All Layers)")
    print("="*80)

    orchestrator = ProbeOrchestrator(llm, backend=backend, revision=llm_revision)

    data = SingleFeatureData(
        prompts=fact_prompts_large,
        labels=fact_labels_large
    )
    print(f"Dataset: {len(fact_prompts_large)} prompts ({sum(fact_labels_large)} true, {len(fact_labels_large) - sum(fact_labels_large)} false)")

    results = orchestrator.probe(
        data=data,
        layers="all",
        position=PositionOption.ALL,
        components=None,
        include_selectivity=True,
        random_trials=2,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    print(f"\nAnalyzing {len(results.results)} probe results across positions...")
    print(f"Best overall: Layer {results.best_layer}, Accuracy {results.best_result.accuracy:.1%}")

    results.plot_heatmap_interactive(output_path="scenario3.html", show=False)
    results.generate_report(output_path="scenario3_report.html", show=False)
    print("✓ Saved: scenario3.html")

    return results


def scenario_4_multi_feature_shared():
    """
    Scenario 4: Probe Multiple Features (Shared Prompts) on Last Token, All Layers

    This is the MOST EFFICIENT mode for probing multiple features on shared prompts:
    - Same prompts (factuality_topic_shared: statements with both factuality and topic labels)
    - Two different label sets: factuality (true=1, false=0) and topic (math=1, climate=0)
    - Activations extracted ONCE and reused
    - Perfect for comparing different ways of categorizing the same data
    """
    print("\n" + "="*80)
    print("SCENARIO 4: Multi-Feature Shared Prompts (Factuality vs Topic)")
    print("="*80)

    orchestrator = ProbeOrchestrator(llm, backend=backend, revision=llm_revision)

    # Shared prompts with two different label dimensions (factuality and topic)
    data = MultiFeatureSharedPromptsData(
        prompts=scenario4_prompts,
        features={
            "factuality": scenario4_factuality_labels,  # true=1, false=0
            "topic": scenario4_topic_labels,            # math=1, climate=0
        }
    )

    print(f"Probing {data.num_features} features on {data.num_samples} shared prompts")
    print(f"Features: {', '.join(data.feature_names)}")
    print(f"Factuality: {sum(scenario4_factuality_labels)} true, {len(scenario4_factuality_labels) - sum(scenario4_factuality_labels)} false")
    print(f"Topic: {sum(scenario4_topic_labels)} math, {len(scenario4_topic_labels) - sum(scenario4_topic_labels)} climate")

    results = orchestrator.probe(
        data=data,
        layers="all",
        position=PositionOption.LAST,
        components=None,
        include_selectivity=True,
        random_trials=2,
        max_workers=max_workers,
        batch_size=batch_size,
        checkpoint_dir="checkpoints/scenario4",
    )

    print("\nFeature comparison:")
    for feature_name in results.feature_names:
        feature_results = results[feature_name]
        print(f"  {feature_name:12s}: Best layer {feature_results.best_layer:2d}, "
              f"Accuracy {feature_results.best_result.accuracy:.1%}, "
              f"Selectivity {feature_results.best_result.selectivity:.1%}")

    results.plot_heatmap_interactive(output_path="scenario4.html", show=False)
    results.generate_report(output_path="scenario4_report.html", show=False)
    print("✓ Saved: scenario4.html")

    return results


def scenario_5_multi_feature_separate():
    """
    Scenario 5: Probe Multiple Features (Separate Prompts) on Last Token, All Layers

    Compare two different features from independent datasets:
    - Factuality detection (factuality_large: true=1, false=0)
    - Topic detection (topics_large: math=1, climate=0)

    This demonstrates probing for completely different features with separate prompts.
    """
    print("\n" + "="*80)
    print("SCENARIO 5: Multi-Feature Separate Prompts (Factuality vs Topic)")
    print("="*80)

    orchestrator = ProbeOrchestrator(llm, backend=backend, revision=llm_revision)

    # Two independent datasets: factuality and topics
    data = MultiFeatureSeparatePromptsData(
        features={
            "factuality": (fact_prompts_large, fact_labels_large),    # true=1, false=0
            "topic": (topics_prompts_large, topics_labels_large),      # math=1, climate=0
        }
    )

    print(f"Probing {data.num_features} features with separate prompts")
    for feature_name in data.feature_names:
        feature_prompts, labels = data.get_feature_data(feature_name)
        positive_count = sum(labels)
        print(f"  {feature_name}: {len(feature_prompts)} prompts ({positive_count} positive, {len(labels) - positive_count} negative)")

    results = orchestrator.probe(
        data=data,
        layers="all",
        position=PositionOption.LAST,
        components=None,
        include_selectivity=True,
        random_trials=2,
        max_workers=max_workers,
        batch_size=batch_size,
        checkpoint_dir="checkpoints/scenario5",
    )

    print("\nFeature comparison:")
    for feature_name in results.feature_names:
        feature_results = results[feature_name]
        print(f"  {feature_name:12s}: Best layer {feature_results.best_layer:2d}, "
              f"Accuracy {feature_results.best_result.accuracy:.1%}, "
              f"Selectivity {feature_results.best_result.selectivity:.1%}")

    results.plot_heatmap_interactive(output_path="scenario5.html", show=False)
    results.generate_report(output_path="scenario5_report.html", show=False)
    print("✓ Saved: scenario5.html")

    return results


def scenario_6_model_comparison():
    """
    Scenario 6: Compare OLMo-3 Across Training Stages

    Compare how factuality detection evolves at different training stages:
    - Stage 1: Initial pretraining (5.93T tokens on Dolma 3)
    - Stage 2: Mid-training (100B tokens on Dolmino-mix)
    - Stage 3: Long context training (50B tokens on Longmino-mix)

    Uses NNSight backend for OLMo-3 model support.

    Use this to answer: "How does factuality encoding evolve during training?"
    """
    print("\n" + "="*80)
    print("SCENARIO 6: OLMo-3 Training Stage Comparison")
    print("="*80)
    print(f"Base model: {OLMO3_BASE_MODEL}")
    print(f"Comparing {len(OLMO3_STAGES)} training stages:")
    for revision, stage_name in OLMO3_STAGES:
        print(f"  - {stage_name}: {revision}")

    # Prepare data (same for all stages) - detect factuality
    data = SingleFeatureData(
        prompts=fact_prompts_large,
        labels=fact_labels_large
    )
    print(f"Dataset: {len(fact_prompts_large)} prompts ({sum(fact_labels_large)} true, {len(fact_labels_large) - sum(fact_labels_large)} false)")

    results_dict = {}
    checkpoint_dirs = []  # Track checkpoint dirs for cleanup after reports

    # Probe each training stage
    for revision, stage_name in OLMO3_STAGES:
        print(f"\n--- Probing {stage_name} ({revision}) ---")

        # Create orchestrator with NNSight backend and specific revision
        orchestrator = ProbeOrchestrator(
            OLMO3_BASE_MODEL,
            backend=BackendOption.NNSIGHT,
            revision=revision,
        )

        # Use revision as checkpoint dir name (replace slashes)
        checkpoint_name = revision.replace("/", "_").replace("-", "_")
        checkpoint_dir = f"checkpoints/scenario6_{checkpoint_name}"
        checkpoint_dirs.append(checkpoint_dir)

        # Don't auto-cleanup checkpoints - we'll clean up after all reports are generated
        results = orchestrator.probe(
            data=data,
            layers="all",
            position=PositionOption.LAST,
            components=None,
            include_selectivity=True,
            random_trials=2,
            max_workers=max_workers,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            auto_cleanup=False,  # Keep checkpoints until all stages complete
        )
        results_dict[stage_name] = results

    # Compare results
    print("\n" + "-"*60)
    print("TRAINING STAGE COMPARISON RESULTS")
    print("-"*60)

    for stage_name, results in results_dict.items():
        n_layers = len(set(r.layer for r in results.results))
        print(f"\n{stage_name}:")
        print(f"  Layers: {n_layers}")
        print(f"  Best layer: {results.best_layer}")
        print(f"  Best accuracy: {results.best_result.accuracy:.1%}")
        if results.best_result.selectivity is not None:
            print(f"  Selectivity: {results.best_result.selectivity:.1%}")

    # Save combined heatmap (all stages on same HTML)
    plot_multi_model_heatmap(
        results_dict,
        title="OLMo-3 Training Stages - Factuality Probes",
        output_path="scenario6.html",
        show=False,
    )
    print("\n✓ Saved: scenario6.html")

    # Generate combined report (all stages in one report)
    generate_multi_model_report(
        results_dict,
        output_path="scenario6_report.html",
        show=False,
    )
    print("✓ Saved: scenario6_report.html")

    # Clean up checkpoint directories now that all reports are generated
    print("\n[EasyProbe] Cleaning up checkpoint directories...")
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            print(f"  Removed: {checkpoint_dir}")

    return results_dict


def main():
    """Run all 6 scenarios."""
    print("\n" + "="*80)
    print("EASYPROBE: Factuality Probing Demonstration")
    print("="*80)
    print(f"Factuality dataset: {len(fact_prompts_large)} prompts ({sum(fact_labels_large)} true, {len(fact_labels_large) - sum(fact_labels_large)} false)")
    print(f"Topics dataset: {len(topics_prompts_large)} prompts ({sum(topics_labels_large)} math, {len(topics_labels_large) - sum(topics_labels_large)} climate)")
    print(f"Multi-label dataset: {len(scenario4_prompts)} prompts (factuality + topic)")
    print(f"Model (scenarios 1-5): {llm}")
    print(f"Model (scenario 6): {OLMO3_BASE_MODEL} (3 training stages)")
    print(f"Backend (scenarios 1-5): {backend.value}")

    # Run each scenario
    try:
        scenario_1_basic_layer_sweep()
        clear_gpu_memory()

        scenario_2_component_comparison()
        clear_gpu_memory()

        scenario_3_position_analysis()
        clear_gpu_memory()

        scenario_4_multi_feature_shared()
        clear_gpu_memory()

        scenario_5_multi_feature_separate()
        clear_gpu_memory()

        scenario_6_model_comparison()

        print("\n" + "="*80)
        print("✓ All scenarios completed successfully!")
        print("="*80)
        print("\nGenerated files:")
        print("  - scenario1.html, scenario1_report.html (layer sweep)")
        print("  - scenario2.html, scenario2_report.html (component comparison)")
        print("  - scenario3.html, scenario3_report.html (position analysis)")
        print("  - scenario4.html, scenario4_report.html (multi-feature shared)")
        print("  - scenario5.html, scenario5_report.html (multi-feature separate)")
        print("  - scenario6.html, scenario6_report.html (OLMo-3 training stages)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

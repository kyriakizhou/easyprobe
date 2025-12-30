"""
EasyProbe: Comprehensive demonstration of all use scenarios.

This script demonstrates 6 key scenarios for linear probing with easyprobe:
1. Single feature, last token, all layers (basic layer sweep)
2. Single feature, last token, all components (component comparison)
3. Single feature, all positions, all layers (position analysis)
4. Multiple features (shared prompts), last token, all layers (multi-feature efficiency)
5. Multiple features (separate prompts), last token, all layers (independent features)
6. OLMo-3 training stages comparison (pre-training, mid-training, long-context)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from easyprobe import ProbeOrchestrator, SingleFeatureData, MultiFeatureSharedPromptsData, MultiFeatureSeparatePromptsData
from easyprobe.datamodels import ComponentOption, PositionOption, BackendOption
from easyprobe.visualization import plot_multi_model_heatmap, generate_multi_model_report
from easyprobe.data.expertise import (
    expertise_prompts,
    expertise_novice_labels,
    expertise_medium_labels,
    expertise_expert_labels,
)
from easyprobe.data.gender import gender_prompts, gender_labels

TEST_CORRECTNESS = True
USE_NNSIGHT = True  # Set to True to use NNSight backend, False for TransformerLens

# Backend configuration
if USE_NNSIGHT:
    # Use gpt2 (124M) for fast local testing, Qwen2.5-3B for production
    # gpt2 is much smaller and faster than Qwen2.5-0.5B (500M)
    llm = "gpt2" if TEST_CORRECTNESS else "Qwen/Qwen2.5-3B"
    backend = BackendOption.NNSIGHT
else:
    llm = "pythia-70m" if TEST_CORRECTNESS else "Qwen/Qwen2.5-3B"
    backend = BackendOption.TRANSFORMERLENS

# Expertise data setup (one-vs-rest strategy)
# All scenarios use the same prompts but different binary labels:
# - novice_labels: novice=1, others=0
# - medium_labels: medium=1, others=0
# - expert_labels: expert=1, others=0
prompts = expertise_prompts
novice_labels = expertise_novice_labels
medium_labels = expertise_medium_labels
expert_labels = expertise_expert_labels

# Default to expert detection for single-feature scenarios
default_labels = expert_labels
default_feature_name = "expert"

# Model comparison settings
llm_comparison_1 = "pythia-70m"
llm_comparison_2 = "pythia-160m"
max_workers = 7
batch_size = 4 if TEST_CORRECTNESS else 16  # Smaller batch for local testing (MPS memory)

# OLMo-3 training stages for scenario 6 (uses NNSight backend)
# These are the 3 training stages with their final checkpoints:
# - Stage 1: Initial pretraining (5.93T tokens on Dolma 3)
# - Stage 2: Mid-training (100B tokens on Dolmino-mix)
# - Stage 3: Long context training (50B tokens on Longmino-mix)
OLMO3_BASE_MODEL = "allenai/Olmo-3-1025-7B"
OLMO3_STAGES = [
    ("stage1-step999000", "Stage 1 (Pretraining)"),
    ("stage2-step9000", "Stage 2 (Mid-training)"),
    ("stage3-step9000", "Stage 3 (Long Context)"),
]


def scenario_1_basic_layer_sweep():
    """
    Scenario 1: Probe Single Feature on Last Token Position of All Layers

    This is the most basic and common use case:
    - One feature (expertise: expert vs others)
    - Last token position (standard for autoregressive models)
    - All layers (find where the feature emerges)

    Use this to answer: "At which layer does the model encode expertise level?"
    """
    print("\n" + "="*80)
    print("SCENARIO 1: Basic Layer Sweep (Expert Detection, Last Token, All Layers)")
    print("="*80)

    # Create orchestrator
    orchestrator = ProbeOrchestrator(llm, backend=backend)

    # Prepare data - detect expert conversations (expert=1, others=0)
    data = SingleFeatureData(
        prompts=prompts,
        labels=expert_labels
    )

    # Run probe
    results = orchestrator.probe(
        data=data,
        layers="all",                      # Probe all layers
        position=PositionOption.LAST,      # Last token only
        components=None,                   # Residual stream only (default)
        include_selectivity=True,          # Compute random baseline
        random_trials=2,                   # 2 trials for selectivity
        max_workers=max_workers,           # Parallel training
        batch_size=batch_size,
    )

    # Visualize results
    print(f"\nBest layer: {results.best_layer}")
    print(f"Best accuracy: {results.best_result.accuracy:.1%}")
    print(f"Selectivity: {results.best_result.selectivity:.1%}")

    # Save plots
    results.plot_heatmap_interactive(output_path="scenario1.html", show=False)
    results.generate_report(output_path="scenario1_report.html", show=False)
    print("✓ Saved: scenario1.html")

    return results


def scenario_2_component_comparison():
    """
    Scenario 2: Probe Single Feature on Last Token Position of All Components

    Compare how expertise is encoded in:
    - Residual stream (cumulative representation)
    - Attention outputs (what the model is "looking at")
    - MLP outputs (non-linear transformations)

    Use this to answer: "Which component encodes expertise level better?"
    """
    print("\n" + "="*80)
    print("SCENARIO 2: Component Comparison (Attention vs MLP vs Residual)")
    print("="*80)

    orchestrator = ProbeOrchestrator(llm, backend=backend)

    data = SingleFeatureData(
        prompts=prompts,
        labels=expert_labels
    )

    # Probe ALL components
    results = orchestrator.probe(
        data=data,
        layers="all",
        position=PositionOption.LAST,
        components=[ComponentOption.RESID, ComponentOption.ATTN, ComponentOption.MLP],  # All components!
        include_selectivity=True,
        random_trials=2,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    # Analyze component differences
    print("\nComponent comparison:")
    for component in [ComponentOption.RESID, ComponentOption.ATTN, ComponentOption.MLP]:
        component_results = [r for r in results.results if r.component == component]
        if component_results:
            best = max(component_results, key=lambda r: r.accuracy)
            print(f"  {component.value:6s}: Layer {best.layer:2d}, Accuracy {best.accuracy:.1%}, Selectivity {best.selectivity:.1%}")

    # Save heatmap
    results.plot_heatmap_interactive(output_path="scenario2.html", show=False)
    results.generate_report(output_path="scenario2_report.html", show=False)
    print("✓ Saved: scenario2.html")

    return results


def scenario_3_position_analysis():
    """
    Scenario 3: Probe Single Feature on All Token Positions of All Layers

    Use this to answer: "Does expertise emerge at specific token positions?"
    """
    print("\n" + "="*80)
    print("SCENARIO 3: Position Analysis (All Tokens, All Layers)")
    print("="*80)

    orchestrator = ProbeOrchestrator(llm, backend=backend)

    data = SingleFeatureData(
        prompts=prompts,
        labels=expert_labels
    )

    # Probe ALL token positions
    results = orchestrator.probe(
        data=data,
        layers="all",
        position=PositionOption.ALL,       # ALL positions!
        components=None,                    # Residual only
        include_selectivity=True,
        random_trials=2,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    print(f"\nAnalyzing {len(results.results)} probe results across positions...")
    print(f"Best overall: Layer {results.best_layer}, Accuracy {results.best_result.accuracy:.1%}")

    # Save position heatmap
    results.plot_heatmap_interactive(output_path="scenario3.html", show=False)
    results.generate_report(output_path="scenario3_report.html", show=False)
    print("✓ Saved: scenario3.html")

    return results


def scenario_4_multi_feature_shared():
    """
    Scenario 4: Probe Multiple Features (Shared Prompts) on Last Token, All Layers

    This is the MOST EFFICIENT mode for one-vs-rest expertise classification:
    - Same prompts, three different label sets (novice, medium, expert)
    - Activations extracted ONCE and reused
    - Perfect for comparing which expertise levels are most distinguishable
    """
    print("\n" + "="*80)
    print("SCENARIO 4: Multi-Feature Shared Prompts (One-vs-Rest Expertise)")
    print("="*80)

    orchestrator = ProbeOrchestrator(llm, backend=backend)

    # Shared prompts with three one-vs-rest label sets
    data = MultiFeatureSharedPromptsData(
        prompts=prompts,
        features={
            "novice": novice_labels,   # novice=1, others=0
            "medium": medium_labels,   # medium=1, others=0
            "expert": expert_labels,   # expert=1, others=0
        }
    )

    print(f"Probing {data.num_features} features on {data.num_samples} shared prompts")
    print(f"Features: {', '.join(data.feature_names)}")

    # Activations will be extracted ONCE
    results = orchestrator.probe(
        data=data,
        layers="all",
        position=PositionOption.LAST,
        components=None,
        include_selectivity=True,
        random_trials=2,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    # Compare features
    print("\nFeature comparison (one-vs-rest):")
    for feature_name in results.feature_names:
        feature_results = results[feature_name]
        print(f"  {feature_name:12s}: Best layer {feature_results.best_layer:2d}, "
              f"Accuracy {feature_results.best_result.accuracy:.1%}, "
              f"Selectivity {feature_results.best_result.selectivity:.1%}")

    # Save comparison plot
    results.plot_heatmap_interactive(output_path="scenario4.html", show=False)
    results.generate_report(output_path="scenario4_report.html", show=False)
    print("✓ Saved: scenario4.html")

    return results


def scenario_5_multi_feature_separate():
    """
    Scenario 5: Probe Multiple Features (Separate Prompts) on Last Token, All Layers

    Compare two different user attributes from independent datasets:
    - Gender detection (from llama_gender data): female=1, male=0
    - Expertise detection (from opus-expertise data): expert=1, others=0

    This demonstrates probing for completely different features with separate prompts.
    """
    print("\n" + "="*80)
    print("SCENARIO 5: Multi-Feature Separate Prompts (Gender vs Expertise)")
    print("="*80)

    orchestrator = ProbeOrchestrator(llm, backend=backend)

    # Two independent datasets: gender and expertise
    data = MultiFeatureSeparatePromptsData(
        features={
            "gender": (gender_prompts, gender_labels),        # female=1, male=0
            "expertise": (expertise_prompts, expert_labels),  # expert=1, others=0
        }
    )

    print(f"Probing {data.num_features} features with separate prompts")
    for feature_name in data.feature_names:
        feature_prompts, labels = data.get_feature_data(feature_name)
        positive_count = sum(labels)
        print(f"  {feature_name}: {len(feature_prompts)} prompts ({positive_count} positive, {len(labels) - positive_count} negative)")

    # Each feature gets its own activation extraction
    results = orchestrator.probe(
        data=data,
        layers="all",
        position=PositionOption.LAST,
        components=None,
        include_selectivity=True,
        random_trials=2,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    # Compare results
    print("\nFeature comparison:")
    for feature_name in results.feature_names:
        feature_results = results[feature_name]
        print(f"  {feature_name:12s}: Best layer {feature_results.best_layer:2d}, "
              f"Accuracy {feature_results.best_result.accuracy:.1%}, "
              f"Selectivity {feature_results.best_result.selectivity:.1%}")

    # Save comparison
    results.plot_heatmap_interactive(output_path="scenario5.html", show=False)
    results.generate_report(output_path="scenario5_report.html", show=False)
    print("✓ Saved: scenario5.html")

    return results


def scenario_6_model_comparison():
    """
    Scenario 6: Compare OLMo-3 Across Training Stages

    Compare how expertise detection evolves at different training stages:
    - Stage 1: Initial pretraining (5.93T tokens on Dolma 3)
    - Stage 2: Mid-training (100B tokens on Dolmino-mix)
    - Stage 3: Long context training (50B tokens on Longmino-mix)

    Uses NNSight backend for OLMo-3 model support.

    Use this to answer: "How does expertise encoding evolve during training?"
    """
    print("\n" + "="*80)
    print("SCENARIO 6: OLMo-3 Training Stage Comparison")
    print("="*80)
    print(f"Base model: {OLMO3_BASE_MODEL}")
    print(f"Comparing {len(OLMO3_STAGES)} training stages:")
    for revision, stage_name in OLMO3_STAGES:
        print(f"  - {stage_name}: {revision}")

    # Prepare data (same for all stages) - detect expert conversations
    data = SingleFeatureData(
        prompts=prompts,
        labels=expert_labels
    )

    results_dict = {}

    # Probe each training stage
    for revision, stage_name in OLMO3_STAGES:
        print(f"\n--- Probing {stage_name} ({revision}) ---")

        # Create orchestrator with NNSight backend and specific revision
        orchestrator = ProbeOrchestrator(
            OLMO3_BASE_MODEL,
            backend=BackendOption.NNSIGHT,
            revision=revision,
        )

        results = orchestrator.probe(
            data=data,
            layers="all",
            position=PositionOption.LAST,
            components=None,
            include_selectivity=True,
            random_trials=2,
            max_workers=max_workers,
            batch_size=16,
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
        title="OLMo-3 Training Stages - Expertise Probes",
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

    return results_dict


def main():
    """Run all 6 scenarios."""
    print("\n" + "="*80)
    print("EASYPROBE: Expertise Level Probing Demonstration")
    print("="*80)
    print(f"Dataset: {len(prompts)} conversations (150 novice, 150 medium, 150 expert)")
    print(f"Model (scenarios 1-5): {llm}")
    print(f"Model (scenario 6): {OLMO3_BASE_MODEL} (3 training stages)")
    print(f"Backend (scenarios 1-5): {backend.value}")

    # Run each scenario
    try:
        scenario_1_basic_layer_sweep()
        scenario_2_component_comparison()
        # scenario_3_position_analysis()
        scenario_4_multi_feature_shared()
        scenario_5_multi_feature_separate()
        # scenario_6_model_comparison()

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

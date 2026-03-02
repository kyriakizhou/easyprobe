"""
This script demonstrates 6 important scenarios for linear probing with easyprobe:
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
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers
for _lib in ("httpx", "httpcore", "huggingface_hub", "transformers", "nnsight"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from easyprobe import ProbeOrchestrator, SingleFeatureData, MultiFeatureSharedPromptsData, MultiFeatureSeparatePromptsData
from easyprobe.models.data_models import ComponentOption, PositionOption, BackendOption, AggregationMethod, LayerOption
from easyprobe.models.steering import NNSightSteeringContext
from easyprobe.visualization import plot_multi_model_heatmap, generate_multi_model_report, get_model_comparison_string
from easyprobe.visualization.text_highlight import generate_highlight_map_from_results
from easyprobe.data.factuality import (
    fact_prompts_large,
    fact_labels_large,
    topics_prompts_large,
    topics_labels_large,
    scenario4_prompts,
    scenario4_factuality_labels,
    scenario4_topic_labels,
)

TEST_CORRECTNESS = True # True for local testing with small models, False for bigger / production level models
USE_NNSIGHT = False  # True to use NNSight backend, False for TransformerLens

if USE_NNSIGHT:
    BACKEND = BackendOption.NNSIGHT
else: 
    BACKEND = BackendOption.TRANSFORMERLENS

if TEST_CORRECTNESS:
    LLM = "gpt2"
    LLM_REVISION = None
else:
    LLM = "allenai/Olmo-3-1025-7B"
    LLM_REVISION = "stage3-step9000" # Final stage (Long Context)

MAX_WORKERS = 11 # number of workers for probe training in parallel
BATCH_SIZE = 2 if TEST_CORRECTNESS else 1  # batch_size=1 for OLMo-3 7B on A100 (long conversations use lots of memory)

# OLMo-3 training stages for scenario 6
if TEST_CORRECTNESS:
    LLM_TRAINING_STAGES = [("main", "gpt2 main")]
else:
    LLM_TRAINING_STAGES = [
        ("stage1-step1413814", "Stage 1 (Pretraining)"),
        ("stage2-step47684", "Stage 2 (Mid-training)"),
        ("stage3-step11921", "Stage 3 (Long Context)"),
    ]


def _clear_gpu_memory():
    """Clear GPU memory between scenarios."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def scenario_1_basic_layer_sweep():
    """
    Scenario 1: Probe Single Feature on Last Token Position of All Layers

    This demonstrates:
    - Train probes
    - AUC metric for ranking quality
    - Save probe weights to disk
    - Classify new examples with predict()
    - Steer the model using the probe

    Use this to answer: "At which layer does the model encode factuality?"
    """
    logger.info("SCENARIO 1: Basic Layer Sweep (Factuality Detection, Last Token, All Layers)")

    orchestrator = ProbeOrchestrator(LLM, backend=BACKEND, revision=LLM_REVISION)

    # Prepare data
    data = SingleFeatureData(
        prompts=fact_prompts_large,
        labels=fact_labels_large
    )
    logger.debug(data.get_dataset_info_string())

    # 1. Train probes
    logger.info("[1/5] Training probes across all layers...")
    results = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.LAST,
        include_selectivity=True,
        random_trials=2,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
        activation_checkpoint_path="checkpoints/scenario1",
    )

    # 2. Display metrics
    logger.info(results.get_metrics_string())

    # 3. Save all probe weights and full results object for later analysis/visualizations
    logger.info("[2/5] Saving probe weights...")
    results.save_probes(path="trained_probes/scenario1")
    results.save("checkpoints/scenario1/results.pkl")
    logger.debug(f"✓ Saved {len(results.trained_probes)} probes to trained_probes/scenario1/ and full results object to checkpoints/scenario1/results.pkl")

    # 4. Classify new examples using predict()
    logger.info("[3/5] Classifying new examples with best probe...")
    
    test_examples = [
        # True statements
        "Water consists of two hydrogen atoms and one oxygen atom",
        "Light travels faster than sound",
        "Photosynthesis is how plants convert sunlight into energy",
        "Humans need to breathe oxygen to survive",
        "Diamonds are one of the hardest natural materials",
        # False statements
        "The moon is made entirely of green cheese",
        "Fish are mammals that breathe through lungs",
        "The Sun revolves around the Earth once a day",
        "Humans can survive without drinking water for several years",
        "Paris is the capital of Germany"
    ]
    
    best_probe = results.best_result
    class_results = []
    for text in test_examples:
        score, label = orchestrator.predict(
            text=text, 
            probe=best_probe,
            aggregation=AggregationMethod.MEAN,
            threshold=0.5
        )
        class_results.append(f"'{text}' → Score: {score:.3f}, Predicted: {'TRUE' if label == 1 else 'FALSE'}")
    logger.info("Classification predictions:\n  " + "\n  ".join(class_results))

    # 5. Steer the model and generate text
    logger.info("[4/5] Steering model with factuality probe...")
    steer_prompt = "The capital of France is"
    max_gen_tokens = 15
    
    # Get the underlying model
    model = orchestrator.extractor.model
    
    # --- Generate WITHOUT steering ---
    if USE_NNSIGHT:
        no_steer_text = NNSightSteeringContext.generate_text(model, steer_prompt, max_new_tokens=max_gen_tokens)
    else:
        tokens = model.to_tokens(steer_prompt)
        output = model.generate(tokens, max_new_tokens=max_gen_tokens, temperature=0.7)
        no_steer_text = model.to_string(output[0])
    
    # --- Generate WITH steering (amplify factuality) ---
    steer_multiplier = 3.0
    
    # Standard Steering
    steering_ctx = best_probe.steer(model, multiplier=steer_multiplier)
    steered_text = steering_ctx.generate(steer_prompt, max_new_tokens=max_gen_tokens)
    
    # Dual Steering (only supported for NNSight)
    if USE_NNSIGHT:
        dual_steering_ctx = best_probe.steer(model, multiplier=steer_multiplier, method="dual")
        dual_steered_text = dual_steering_ctx.generate(steer_prompt, max_new_tokens=max_gen_tokens)
        
        logger.info(f"Steering results for prompt '{steer_prompt}':\n  Without steering: {no_steer_text}\n  With standard steering (mult={steer_multiplier}): {steered_text}\n  With dual steering (mult={steer_multiplier}): {dual_steered_text}")
    else:
        logger.info(f"Steering results for prompt '{steer_prompt}':\n  Without steering: {no_steer_text}\n  With steering (mult={steer_multiplier}): {steered_text}")

    # 6. Generate visualizations
    logger.info("[5/5] Generating visualizations...")
    results.plot_heatmap_interactive(path="scenario1.html", show=False)
    results.generate_report(path="scenario1_report.html", show=False)
    logger.info("✓ Saved visualizations: scenario1.html, scenario1_report.html")

    return results



def scenario_2_component_comparison():
    """
    Scenario 2: Probe Single Feature on Last Token Position of All Components

    Compare how factuality is encoded in:
    - Residual stream (cumulative representation)
    - Attention outputs (before adding back to resid)
    - MLP outputs (before adding back to resid)

    Use this to answer: "Which component encodes factuality better?"
    """
    logger.info("SCENARIO 2: Component Comparison (Attention vs MLP vs Residual)")

    orchestrator = ProbeOrchestrator(LLM, backend=BACKEND, revision=LLM_REVISION)

    data = SingleFeatureData(
        prompts=fact_prompts_large,
        labels=fact_labels_large
    )
    logger.info(data.get_dataset_info_string())

    results = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.LAST,
        components=[ComponentOption.RESID, ComponentOption.ATTN, ComponentOption.MLP],
        include_selectivity=True,
        random_trials=2,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
        activation_checkpoint_path="checkpoints/scenario2",
    )

    logger.info(results.get_component_comparison_string())

    results.plot_heatmap_interactive(path="scenario2.html", show=False)
    results.generate_report(path="scenario2_report.html", show=False)
    logger.info("✓ Saved visualizations: scenario2.html, scenario2_report.html")

    return results


def scenario_3_position_analysis():
    """
    Scenario 3: Probe Single Feature on All Token Positions of All Layers
    """
    logger.info("SCENARIO 3: Position Analysis (All Tokens, All Layers)")

    orchestrator = ProbeOrchestrator(LLM, backend=BACKEND, revision=LLM_REVISION)

    data = SingleFeatureData(
        prompts=fact_prompts_large,
        labels=fact_labels_large
    )
    logger.info(data.get_dataset_info_string())

    results = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.ALL,
        components=None,
        include_selectivity=True,
        random_trials=2,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
    )

    logger.info(f"Analyzed {len(results.trained_probes)} probe results across positions. Best overall: Layer {results.best_layer}, Accuracy {results.best_result.accuracy:.1%}")

    results.plot_heatmap_interactive(path="scenario3.html", show=False)
    results.generate_report(path="scenario3_report.html", show=False)
    logger.info("✓ Saved visualizations: scenario3.html, scenario3_report.html")

    # Generate text highlight map using position-specific probes
    sample_text = "Water consists of two hydrogen atoms and one oxygen atom."
    
    # We use "best_token_layer" to find the most accurate probe across model layers for each individual token position
    generate_highlight_map_from_results(
        results=results,
        orchestrator=orchestrator,
        text=sample_text,
        method="best_token_layer",  # You can toggle this to "best_global_layer" if needed
        save_path="scenario3_highlight.html"
    )

    return results


def scenario_4_multi_feature_shared():
    """
    Scenario 4: Probe Multiple Features (Shared Prompts) on Last Token, All Layers

    This is the MOST EFFICIENT mode for probing multiple features on shared prompts:
    - Same prompts (factuality_topic_shared: statements with both factuality and topic concept encoded)
    - Two different label sets: factuality (true=1, false=0) and topic (math=1, climate=0)
    - Activations extracted ONCE and reused
    - Perfect for comparing different ways of categorizing the same data
    """
    logger.info("SCENARIO 4: Multi-Feature Shared Prompts (Factuality vs Topic)")

    orchestrator = ProbeOrchestrator(LLM, backend=BACKEND, revision=LLM_REVISION)

    data = MultiFeatureSharedPromptsData(
        prompts=scenario4_prompts,
        features={
            "factuality": scenario4_factuality_labels, 
            "topic": scenario4_topic_labels,            
        }
    )
    logger.info(data.get_dataset_info_string())

    results = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.LAST,
        include_selectivity=True,
        random_trials=2,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
        activation_checkpoint_path="checkpoints/scenario4",
    )

    logger.info(results.get_feature_comparison_string())

    results.plot_heatmap_interactive(path="scenario4.html", show=False)
    results.generate_report(path="scenario4_report.html", show=False)
    logger.info("✓ Saved visualizations: scenario4.html, scenario4_report.html")

    return results


def scenario_5_multi_feature_separate():
    """
    Scenario 5: Probe Multiple Features (Separate Prompts) on Last Token, All Layers
    This is logically equivalent to running Scenario 1 sequentially for each feature
    but I still want to provide this as a use case to reduce boilerplat code and 
    consolidated visualization plots.

    Compare two different features from independent datasets:
    - Factuality detection (factuality_large: true=1, false=0)
    - Topic detection (topics_large: math=1, climate=0)

    This demonstrates probing for completely different features with separate prompts.
    """
    logger.info("SCENARIO 5: Multi-Feature Separate Prompts (Factuality vs Topic)")

    orchestrator = ProbeOrchestrator(LLM, backend=BACKEND, revision=LLM_REVISION)

    # Two independent datasets: factuality and topics
    data = MultiFeatureSeparatePromptsData(
        features={
            "factuality": (fact_prompts_large, fact_labels_large),    # true=1, false=0
            "topic": (topics_prompts_large, topics_labels_large),      # math=1, climate=0
        }
    )
    logger.info(data.get_dataset_info_string())

    results = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.LAST,
        components=None,
        include_selectivity=True,
        random_trials=2,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
        activation_checkpoint_path="checkpoints/scenario5",
    )

    logger.info(results.get_feature_comparison_string())

    results.plot_heatmap_interactive(path="scenario5.html", show=False)
    results.generate_report(path="scenario5_report.html", show=False)
    logger.info("✓ Saved visualizations: scenario5.html, scenario5_report.html")

    return results


def scenario_6_model_comparison():
    """
    Scenario 6: Compare OLMo-3 Across Training Stages

    Compare how factuality detection evolves at different training stages:
    - Stage 1: Initial pretraining (5.93T tokens on Dolma 3)
    - Stage 2: Mid-training (100B tokens on Dolmino-mix)
    - Stage 3: Long context training (50B tokens on Longmino-mix)

    Use this to answer: "How does factuality encoding evolve during training?"
    """
    logger.info("SCENARIO 6: OLMo-3 Training Stage Comparison")
    
    stage_logs = [f"Base model: {LLM} | Comparing {len(LLM_TRAINING_STAGES)} training stages:"]
    for revision, stage_name in LLM_TRAINING_STAGES:
        stage_logs.append(f"- {stage_name}: {revision}")
    logger.info("\n  ".join(stage_logs))

    data = SingleFeatureData(
        prompts=fact_prompts_large,
        labels=fact_labels_large
    )
    logger.info(data.get_dataset_info_string())

    orchestrator = ProbeOrchestrator(
        LLM,
        backend=BackendOption.NNSIGHT,
        revisions=LLM_TRAINING_STAGES,
    )

    results_dict = orchestrator.probe(
        data=data,
        layers=LayerOption.ALL,
        position=PositionOption.LAST,
        components=None,
        include_selectivity=True,
        random_trials=2,
        max_workers=MAX_WORKERS,
        batch_size=BATCH_SIZE,
        activation_checkpoint_path="checkpoints/scenario6",
        auto_cleanup=True,
    )

    # Compare results
    logger.info(get_model_comparison_string(results_dict))

    # Save combined heatmap
    plot_multi_model_heatmap(
        results_dict,
        title="OLMo-3 Training Stages - Factuality Probes",
        path="scenario6.html",
        show=False,
    )
    # Generate combined report
    generate_multi_model_report(
        results_dict,
        path="scenario6_report.html",
        show=False,
    )
    logger.info("✓ Saved visualizations: scenario6.html, scenario6_report.html")

    return results_dict


def main():
    """Run all 6 scenarios."""
    logger.info("EASYPROBE: Factuality Probing Demonstration")
    
    config_logs = [
        f"Factuality dataset: {len(fact_prompts_large)} prompts ({sum(fact_labels_large)} true, {len(fact_labels_large) - sum(fact_labels_large)} false)",
        f"Topics dataset: {len(topics_prompts_large)} prompts ({sum(topics_labels_large)} math, {len(topics_labels_large) - sum(topics_labels_large)} climate)",
        f"Multi-label dataset: {len(scenario4_prompts)} prompts (factuality + topic)",
        f"Model (scenarios 1-5): {LLM}",
        f"Model (scenario 6): {LLM} ({len(LLM_TRAINING_STAGES)} training stages)",
        f"Backend (scenarios 1-5): {BACKEND.value}"
    ]
    logger.info("\n  ".join(config_logs))

    # Run each scenario
    try:
        scenario_1_basic_layer_sweep()
        _clear_gpu_memory()

        scenario_2_component_comparison()
        _clear_gpu_memory()

        scenario_3_position_analysis()
        _clear_gpu_memory()

        scenario_4_multi_feature_shared()
        _clear_gpu_memory()

        scenario_5_multi_feature_separate()
        _clear_gpu_memory()

        scenario_6_model_comparison()

        msg = "✓ All scenarios completed successfully!"
        msg += "\nGenerated files:"
        msg += "\n  - scenario1.html, scenario1_report.html (layer sweep)"
        msg += "\n  - scenario2.html, scenario2_report.html (component comparison)"
        msg += "\n  - scenario3.html, scenario3_report.html (position analysis)"
        msg += "\n  - scenario4.html, scenario4_report.html (multi-feature shared)"
        msg += "\n  - scenario5.html, scenario5_report.html (multi-feature separate)"
        msg += "\n  - scenario6.html, scenario6_report.html (OLMo-3 training stages)"
        logger.info(msg)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

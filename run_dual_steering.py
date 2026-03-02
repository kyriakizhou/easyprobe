import logging
import os
import sys
import glob
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Path Fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# ----------------

from easyprobe import ProbeOrchestrator
from easyprobe.models.data_models import BackendOption
from easyprobe.models.linear_probe import LinearProbe
from easyprobe.models.steering import NNSightSteeringContext

def load_best_probe_from_folder(folder):
    """Finds and loads the probe with the highest accuracy from a folder."""
    probe_files = glob.glob(os.path.join(folder, "*.pkl"))
    if not probe_files:
        raise FileNotFoundError(f"No probe files found in {folder}")
    best_probe = None
    best_acc = -1.0
    for f in probe_files:
        probe = LinearProbe.load(f)
        if probe.accuracy > best_acc:
            best_acc = probe.accuracy
            best_probe = probe
    return best_probe

def main():
    # --- Configuration ---
    llm = "allenai/Olmo-3-1025-7B"
    llm_revision = "stage3-step9000" 
    backend = BackendOption.NNSIGHT 
    probe_folder = "saved_probes/scenario1"

    logger.info(f"Initializing orchestrator for Dual Steering Comparison: {llm}")
    orchestrator = ProbeOrchestrator(llm, backend=backend, revision=llm_revision)
    model = orchestrator.extractor.model
    
    logger.info(f"Searching for best probe in {probe_folder}")
    best_probe = load_best_probe_from_folder(probe_folder)
    logger.info(f"Loaded best probe: Layer {best_probe.layer}, Component {best_probe.component.value}")

    steer_prompt = "The capital of France is"
    max_gen_tokens = 30
    
    logger.info(f"Prompt: '{steer_prompt}'")
    
    # --- 1. Baseline ---
    logger.info("Generating baseline (no steering)...")
    no_steer_text = NNSightSteeringContext.generate_text(model, steer_prompt, max_new_tokens=max_gen_tokens)
    logger.info(f"[Baseline] {no_steer_text}")
    
    # --- 2. Standard Steering ---
    std_multiplier = 4.0
    logger.info(f"Generating with standard steering (multiplier={std_multiplier})...")
    std_ctx = best_probe.steer(model, multiplier=std_multiplier, method="standard")
    std_text = std_ctx.generate(steer_prompt, max_new_tokens=max_gen_tokens)
    logger.info(f"[Standard] {std_text}")
    
    # --- 3. Dual Steering (New Method) ---
    dual_multiplier = 4.0
    logger.info(f"Generating with dual steering (iterations=3, multiplier={dual_multiplier})...")
    dual_ctx = best_probe.steer(
        model, 
        multiplier=dual_multiplier, 
        method="dual", 
        iterations=3, 
        lambda_reg=0.1,
        top_k_cov=100
    )
    dual_text = dual_ctx.generate(steer_prompt, max_new_tokens=max_gen_tokens)
    logger.info(f"[Dual] {dual_text}")

if __name__ == "__main__":
    main()

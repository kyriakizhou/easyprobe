# EasyProbe Development Guide

This guide provides the necessary commands and instructions to run EasyProbe locally on your machine, as well as how to run the provided notebooks on a virtual GPU instance like RunPod.

## Running `main.py` Locally

To run the main execution script on your local machine using your existing Python virtual environment, follow these steps:

1. **Activate the virtual environment**:
   Make sure you are in the root directory of the `easyprobe` project.
   ```bash
   source .venv/bin/activate
   ```

2. **Install or update project dependencies** (if you haven't already):
   Since the project uses a `pyproject.toml`, you can install it in editable mode.
   ```bash
   pip install -e .
   ```

3. **Run the script**:
   ```bash
   python main.py
   ```
   *Note: Add any necessary command-line arguments needed by `main.py` depending on your current task.*

---

## Running `run_easyprobe.ipynb` on a Virtual GPU (e.g., RunPod)

The notebook `notebooks/runpod/run_easyprobe.ipynb` is designed for a quick, self-contained run on a virtual GPU pod (e.g., RunPod). Open it in the pod's built-in JupyterLab and step through the cells:

### 1. Check GPU

Verify that your pod has a working GPU before proceeding:

```python
!nvidia-smi
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

> ⚠️ **Stop here if CUDA says False.** Restart your pod from the RunPod dashboard.

### 2. Upload & Unzip

Upload `easyprobe_runpod.zip` to the notebook directory, then extract it:

```python
!python -c "import zipfile; zipfile.ZipFile('easyprobe_runpod.zip').extractall('easyprobe')"
!ls easyprobe/
```

### 3. Install Dependencies

```python
!pip install -e easyprobe/
!pip install nnsight transformers
```

### 4. Run All Scenarios

```python
!python -m easyprobe.main
```

### 5. Clean Up (Optional)

Delete extracted files and checkpoints before re-running:

```python
!rm -rf easyprobe/ checkpoints/
```

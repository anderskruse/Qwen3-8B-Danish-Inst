# Qwen3-8B Danish Instruct

Fine-tuning Qwen3-8B on Danish instruction data using the skolegpt-instruct dataset.

## Quick Start

```bash
# Install dependencies
uv pip install -e .  # or: pip install -e .

# Login to Hugging Face (need Write token from https://huggingface.co/settings/tokens)
hf auth login

# Train the model
python train.py

# Evaluate
python evaluate.py --model ./outputs/merged_model

# Upload to Hugging Face
python upload_model.py --repo yourusername/Qwen3-8B-Danish-Instruct
```

## Files

- `config.py` - Training hyperparameters
- `train.py` - Main training script
- `evaluate.py` - Test model on Danish prompts
- `upload_model.py` - Upload to Hugging Face Hub

## Training

Uses LoRA fine-tuning with Unsloth.

Default config:
- LoRA r=16, alpha=16
- Learning rate: 1e-4
- Batch size: 4 (with 4x gradient accumulation)
- Epochs: 1

## Dataset

Uses [kobprof/skolegpt-instruct](https://huggingface.co/datasets/kobprof/skolegpt-instruct) - Danish instruction-following dataset.

## RunPod

### 1. Create a pod
- GPU: RTX 4090 or A5000
- Template: PyTorch (runpod/pytorch)
- Volume: 50GB at `/workspace`

### 2. Setup and train
```bash
cd /workspace

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone and install
git clone https://github.com/YOUR_USERNAME/Qwen3-8B-Danish-Inst
cd Qwen3-8B-Danish-Inst
uv pip install --system -e .

# Login to HF
hf auth login

# Start 6h safety timer + train + auto-stop
bash -c "nohup sleep 6h; runpodctl stop pod $RUNPOD_POD_ID" &
python train.py && runpodctl stop pod $RUNPOD_POD_ID
```

### 3. Upload (after pod stops)
```bash
# Start the pod again from dashboard, then:
cd /workspace/Qwen3-8B-Danish-Inst
python upload_model.py --repo YOUR_USERNAME/Qwen3-8B-Danish-Instruct

# Terminate pod after upload is done
```

Weights are saved to `/workspace/Qwen3-8B-Danish-Inst/outputs/merged_model/`.

## License

Apache 2.0

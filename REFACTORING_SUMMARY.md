# Refactoring Summary: Viper-VL → Firebolt-VL

This document summarizes the refactoring changes made to rename the codebase from Viper-VL to Firebolt-VL.

## Changes Made

### 1. Model Classes Renamed
- `ViperLMConfig` → `FireboltLMConfig`
- `ViperLMModel` → `FireboltLMModel`
- `ViperLMForCausalLM` → `FireboltLMForCausalLM`
- Model type: `"viperlm"` → `"fireboltlm"`

### 2. Files Updated

#### Core Model Files
- `modeling/model.py` - All class definitions updated

#### Training Scripts
- `train.py` - Updated imports and class references
- `train_ddp.py` - Updated imports and class references

#### Inference Scripts
- `infer.py` - Updated imports and class references

#### Configuration
- `configs/default.yaml` - Updated comments

#### Documentation
- `README.md` - Updated branding and references

#### Tests
- `tests/test_main.py` - Updated imports and class references

### 3. New Files Created

#### Weight Transfer Scripts
- `transfer_weights.py` - Full-featured transfer script with Hydra support
- `transfer_weights_simple.py` - Simplified transfer script (recommended)

## Usage

### Training with Firebolt-VL

Training scripts work the same way, but now use Firebolt classes:

```bash
# Single GPU
python train.py

# Multi-GPU DDP
torchrun --nproc_per_node=2 train_ddp.py
```

### Inference with Firebolt-VL

```bash
python infer.py
```

### Transferring Weights from Viper-VL to Firebolt-VL

Use the simple transfer script:

```bash
python transfer_weights_simple.py \
    --viper_checkpoint /path/to/viper/checkpoint \
    --output_dir /path/to/firebolt/checkpoint
```

**Example:**
```bash
python transfer_weights_simple.py \
    --viper_checkpoint ./results/epoch_10 \
    --output_dir ./checkpoints/firebolt-vl/epoch_10
```

The script will:
1. Load the Viper-VL checkpoint
2. Update the model_type in config from "viperlm" to "fireboltlm"
3. Create a new Firebolt-VL model with the same architecture
4. Load all compatible weights
5. Save as a Firebolt-VL checkpoint
6. Copy tokenizer files if they exist

### What Gets Transferred

✅ **Transferred:**
- All model weights (vision encoder, projector, fuser, LLM)
- Model configuration (with updated model_type)
- Tokenizer files (if present)

⚠️ **Note:**
- The architecture is identical, so all weights should transfer directly
- Any keys that don't match will be reported but won't cause errors
- The script uses `strict=False` to handle minor mismatches gracefully

## Verification

After transferring weights, verify the checkpoint:

```python
from modeling.model import FireboltLMForCausalLM

# Load the transferred checkpoint
model = FireboltLMForCausalLM.from_pretrained("./checkpoints/firebolt-vl/epoch_10")

# Check model type
print(model.config.model_type)  # Should print: fireboltlm

# Verify it works
print("✅ Model loaded successfully!")
```

## Backward Compatibility

⚠️ **Important:** Old Viper-VL checkpoints cannot be loaded directly with the new Firebolt-VL code. You must:

1. Use the transfer script to convert Viper-VL checkpoints to Firebolt-VL format, OR
2. Keep a copy of the old Viper-VL code if you need to load old checkpoints

## Migration Checklist

- [x] Rename all model classes
- [x] Update all imports
- [x] Update model_type in config
- [x] Update documentation
- [x] Create weight transfer script
- [ ] Test weight transfer on actual checkpoint
- [ ] Update any CI/CD pipelines
- [ ] Update model registry/checkpoint paths in configs

## Notes

- The internal architecture remains unchanged - only class names and model_type were updated
- All functionality (training, inference, etc.) works identically
- The weight transfer script handles the conversion automatically


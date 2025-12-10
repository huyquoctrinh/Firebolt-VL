# Firebolt-VL Quick Start Guide

## Refactoring Complete ✅

The codebase has been successfully refactored from **Viper-VL** to **Firebolt-VL**.

## What Changed

- All model classes renamed: `ViperLM*` → `FireboltLM*`
- Model type updated: `"viperlm"` → `"fireboltlm"`
- All imports and references updated
- Documentation updated

## Using Firebolt-VL

### Training

```bash
# Single GPU
python train.py

# Multi-GPU DDP
torchrun --nproc_per_node=2 train_ddp.py
```

### Inference

```bash
python infer.py
```

## Transferring Viper-VL Weights to Firebolt-VL

If you have existing Viper-VL checkpoints, use the transfer script:

```bash
python transfer_weights_simple.py \
    --viper_checkpoint /path/to/viper/checkpoint \
    --output_dir /path/to/firebolt/checkpoint
```

### Example

```bash
# Transfer a checkpoint
python transfer_weights_simple.py \
    --viper_checkpoint ./results/viper-vl/epoch_10 \
    --output_dir ./checkpoints/firebolt-vl/epoch_10

# Verify it works
python -c "
from modeling.model import FireboltLMForCausalLM
model = FireboltLMForCausalLM.from_pretrained('./checkpoints/firebolt-vl/epoch_10')
print('✅ Model loaded! Type:', model.config.model_type)
"
```

## Important Notes

1. **Old checkpoints**: Viper-VL checkpoints cannot be loaded directly. Use the transfer script first.

2. **Architecture**: The internal architecture is unchanged - only class names changed.

3. **Compatibility**: All training/inference code works identically to before.

## Files to Update

If you have custom configs or scripts, update:
- Model imports: `from modeling.model import FireboltLMForCausalLM, FireboltLMConfig`
- Model type in configs: `model_type: "fireboltlm"`

## Need Help?

See `REFACTORING_SUMMARY.md` for detailed information about all changes.


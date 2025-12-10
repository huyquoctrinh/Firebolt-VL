#!/usr/bin/env python3
"""
Weight Transfer Script: Viper-VL -> Firebolt-VL

This script transfers weights from a Viper-VL checkpoint to a Firebolt-VL model.
It handles:
1. Loading Viper-VL checkpoint
2. Mapping old parameter names to new Firebolt-VL names
3. Saving as Firebolt-VL checkpoint

Usage:
    python transfer_weights.py \
        --viper_checkpoint /path/to/viper/checkpoint \
        --output_dir /path/to/firebolt/checkpoint \
        --config_path configs/default.yaml
"""

import os
import argparse
import torch
import json
from pathlib import Path
from typing import Dict, Any
import hydra
from omegaconf import DictConfig, OmegaConf

# Import both old and new model classes
# We'll use a temporary import for Viper classes
import sys
import importlib.util

# Import new Firebolt classes
from modeling.model import FireboltLMForCausalLM, FireboltLMConfig


def load_viper_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """
    Load Viper-VL checkpoint. We need to temporarily import Viper classes.
    """
    # Dynamically import Viper classes by modifying the model file temporarily
    # Or we can directly load the state dict and config
    checkpoint_path = Path(checkpoint_path)
    
    # Load config
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Check if already Firebolt
    if config_dict.get("model_type") == "fireboltlm":
        print(f"⚠️  Warning: Checkpoint already has model_type='fireboltlm'")
        print(f"   This checkpoint appears to already be a Firebolt-VL checkpoint.")
        print(f"   If you want to convert it anyway, the script will proceed...")
    
    # Update model_type in config
    if config_dict.get("model_type") == "viperlm":
        config_dict["model_type"] = "fireboltlm"
        print(f"   Updated model_type: viperlm -> fireboltlm")
    
    # Load state dict
    model_path = checkpoint_path / "pytorch_model.bin"
    if not model_path.exists():
        # Try safetensors or other formats
        model_path = checkpoint_path / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found in {checkpoint_path}")
    
    if model_path.suffix == ".bin":
        state_dict = torch.load(model_path, map_location=device)
    else:
        # Handle safetensors if needed
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(model_path))
        except ImportError:
            raise ImportError("safetensors library required for .safetensors files")
    
    return state_dict, config_dict


def rename_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Rename state dict keys from Viper-VL to Firebolt-VL naming convention.
    
    Handles common patterns:
    - model.viper_lm_model.* -> model.* (since FireboltLMModel is stored as 'model')
    - model.ViperLMModel.* -> model.*
    - Any Viper/ViperLM references -> Firebolt/FireboltLM
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Check if key contains 'viper' (case-insensitive)
        if 'viper' in key.lower():
            # Handle specific patterns first
            # Pattern 1: model.viper_lm_model.* -> model.*
            if 'model.viper_lm_model.' in key.lower():
                new_key = key.replace('model.viper_lm_model.', 'model.', 1)
                new_key = new_key.replace('model.ViperLMModel.', 'model.', 1)
                print(f"   Renamed: {key[:60]}... -> {new_key[:60]}...")
            # Pattern 2: model.ViperLMModel.* -> model.*
            elif 'model.ViperLMModel.' in key:
                new_key = key.replace('model.ViperLMModel.', 'model.', 1)
                print(f"   Renamed: {key[:60]}... -> {new_key[:60]}...")
            # Pattern 3: General Viper -> Firebolt replacement
            else:
                # Replace 'viper' with 'firebolt' while preserving case
                parts = key.split('.')
                new_parts = []
                for part in parts:
                    if 'viper' in part.lower():
                        # Preserve camelCase: Viper -> Firebolt, viper -> firebolt
                        if part.startswith('Viper'):
                            new_part = part.replace('Viper', 'Firebolt', 1)
                        elif part.startswith('viper'):
                            new_part = part.replace('viper', 'firebolt', 1)
                        elif part.startswith('VIPER'):
                            new_part = part.replace('VIPER', 'FIREBOLT', 1)
                        else:
                            # Mixed case - replace all occurrences
                            new_part = part.replace('Viper', 'Firebolt').replace('viper', 'firebolt')
                        new_parts.append(new_part)
                    else:
                        new_parts.append(part)
                new_key = '.'.join(new_parts)
                if new_key != key:
                    print(f"   Renamed: {key[:60]}... -> {new_key[:60]}...")
        
        new_state_dict[new_key] = value
    
    return new_state_dict


def transfer_weights(
    viper_checkpoint_path: str,
    output_dir: str,
    config: DictConfig = None,
    device: str = "cpu"
):
    """
    Main function to transfer weights from Viper-VL to Firebolt-VL.
    
    Args:
        viper_checkpoint_path: Path to Viper-VL checkpoint directory
        output_dir: Output directory for Firebolt-VL checkpoint
        config: Optional Firebolt config (if None, will use Viper config as base)
        device: Device to load weights on
    """
    print(f"Loading Viper-VL checkpoint from: {viper_checkpoint_path}")
    
    # Load Viper checkpoint
    state_dict, config_dict = load_viper_checkpoint(viper_checkpoint_path, device)
    
    print(f"Loaded state dict with {len(state_dict)} parameters")
    
    # Rename state dict keys
    print("Renaming state dict keys...")
    new_state_dict = rename_state_dict_keys(state_dict)
    
    # Create Firebolt config
    if config is None:
        # Create config from loaded config dict
        # Remove fields that might cause issues (they're handled by PretrainedConfig)
        config_dict_clean = {k: v for k, v in config_dict.items() 
                           if k not in ['architectures', 'transformers_version', 'dtype']}
        try:
            firebolt_config = FireboltLMConfig(**config_dict_clean)
        except TypeError as e:
            print(f"❌ Error creating FireboltLMConfig: {e}")
            print(f"   Config keys: {list(config_dict.keys())}")
            raise
    else:
        firebolt_config = FireboltLMConfig(**config.model)
    
    # Update model_type
    firebolt_config.model_type = "fireboltlm"
    
    # Create Firebolt model
    print("Creating Firebolt-VL model...")
    firebolt_model = FireboltLMForCausalLM(firebolt_config)
    
    # Try to match state dict keys
    # First, try direct loading (most keys should match)
    print("Loading weights into Firebolt-VL model...")
    
    # Get model's expected state dict keys
    model_keys = set(firebolt_model.state_dict().keys())
    state_dict_keys = set(new_state_dict.keys())
    
    # Check key overlap
    matching_keys = model_keys & state_dict_keys
    print(f"Found {len(matching_keys)}/{len(model_keys)} matching keys")
    
    # Load with strict=False to handle any mismatches
    missing_keys, unexpected_keys = firebolt_model.load_state_dict(
        new_state_dict, strict=False
    )
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys:")
        for key in missing_keys[:10]:  # Show first 10
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys:")
        for key in unexpected_keys[:10]:  # Show first 10
            print(f"  - {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more")
    
    # Save Firebolt checkpoint
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving Firebolt-VL checkpoint to: {output_dir}")
    firebolt_model.save_pretrained(str(output_path))
    
    # Also save tokenizer if it exists in source
    source_path = Path(viper_checkpoint_path)
    tokenizer_files = ["tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.json"]
    for tokenizer_file in tokenizer_files:
        src_file = source_path / tokenizer_file
        if src_file.exists():
            import shutil
            dst_file = output_path / tokenizer_file
            shutil.copy2(src_file, dst_file)
            print(f"Copied {tokenizer_file}")
    
    print(f"\n✅ Successfully transferred weights to {output_dir}")
    print(f"   Model type: {firebolt_config.model_type}")
    print(f"   Total parameters: {sum(p.numel() for p in firebolt_model.parameters()):,}")


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    parser = argparse.ArgumentParser(description="Transfer weights from Viper-VL to Firebolt-VL")
    parser.add_argument(
        "--viper_checkpoint",
        type=str,
        required=True,
        help="Path to Viper-VL checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Firebolt-VL checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for loading weights"
    )
    
    args = parser.parse_args()
    
    transfer_weights(
        viper_checkpoint_path=args.viper_checkpoint,
        output_dir=args.output_dir,
        config=cfg,
        device=args.device
    )


if __name__ == "__main__":
    # Allow running without Hydra for simpler usage
    import sys
    if len(sys.argv) > 1 and "--viper_checkpoint" in sys.argv:
        # Parse arguments manually if not using Hydra
        parser = argparse.ArgumentParser(description="Transfer weights from Viper-VL to Firebolt-VL")
        parser.add_argument("--viper_checkpoint", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
        parser.add_argument("--config_path", type=str, default="configs/default.yaml")
        
        args = parser.parse_args()
        
        # Load config if provided
        config = None
        if args.config_path and os.path.exists(args.config_path):
            config = OmegaConf.load(args.config_path)
        
        transfer_weights(
            viper_checkpoint_path=args.viper_checkpoint,
            output_dir=args.output_dir,
            config=config,
            device=args.device
        )
    else:
        # Use Hydra
        main()


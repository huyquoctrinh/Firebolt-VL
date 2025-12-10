#!/usr/bin/env python3
"""
Simple Weight Transfer Script: Viper-VL -> Firebolt-VL

This script transfers weights from a Viper-VL checkpoint to a Firebolt-VL model.
Since the architecture is identical (only class names changed), most weights should
transfer directly.

Usage:
    python transfer_weights_simple.py \
        --viper_checkpoint /path/to/viper/checkpoint \
        --output_dir /path/to/firebolt/checkpoint
"""

import os
import argparse
import torch
import json
from pathlib import Path
from typing import Dict, Any

from modeling.model import FireboltLMForCausalLM, FireboltLMConfig


def rename_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Rename state dict keys from Viper-VL to Firebolt-VL naming convention.
    
    Handles common patterns:
    - model.viper_lm_model.* -> model.* (since FireboltLMModel is stored as 'model')
    - model.ViperLMModel.* -> model.*
    - model.vision_encoder.encoder.model.* -> model.vision_encoder.model.* (remove extra encoder level)
    - Any Viper/ViperLM references -> Firebolt/FireboltLM
    """
    new_state_dict = {}
    renamed_count = 0
    
    for key, value in state_dict.items():
        new_key = key
        was_renamed = False
        
        # Pattern 1: Remove extra .encoder level in vision_encoder
        # Viper: model.vision_encoder.encoder.model.*
        # Firebolt: model.vision_encoder.model.*
        if 'model.vision_encoder.encoder.model.' in key:
            new_key = key.replace('model.vision_encoder.encoder.model.', 'model.vision_encoder.model.', 1)
            was_renamed = True
        
        # Pattern 1b: Add missing .fuser level in fuser (for film/cossm strategy)
        # Viper: model.fuser.model.* or model.fuser.importance.*
        # Firebolt: model.fuser.fuser.model.* or model.fuser.fuser.importance.*
        elif 'model.fuser.model.' in key and 'model.fuser.fuser.model.' not in key:
            new_key = key.replace('model.fuser.model.', 'model.fuser.fuser.model.', 1)
            was_renamed = True
        elif 'model.fuser.importance.' in key and 'model.fuser.fuser.importance.' not in key:
            new_key = key.replace('model.fuser.importance.', 'model.fuser.fuser.importance.', 1)
            was_renamed = True
        
        # Pattern 2: model.viper_lm_model.* -> model.*
        if 'model.viper_lm_model.' in key.lower() and not was_renamed:
            new_key = key.replace('model.viper_lm_model.', 'model.', 1)
            new_key = new_key.replace('model.ViperLMModel.', 'model.', 1)
            was_renamed = True
        
        # Pattern 3: model.ViperLMModel.* -> model.*
        elif 'model.ViperLMModel.' in key and not was_renamed:
            new_key = key.replace('model.ViperLMModel.', 'model.', 1)
            was_renamed = True
        
        # Pattern 4: General Viper -> Firebolt replacement
        elif 'viper' in key.lower() and not was_renamed:
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
                    was_renamed = True
                else:
                    new_parts.append(part)
            new_key = '.'.join(new_parts)
        
        if was_renamed:
            renamed_count += 1
        
        new_state_dict[new_key] = value
    
    if renamed_count > 0:
        print(f"   🔄 Renamed {renamed_count} keys from Viper to Firebolt")
    
    return new_state_dict


def load_checkpoint(checkpoint_path: str):
    """Load checkpoint config and state dict."""
    checkpoint_path = Path(checkpoint_path)
    
    # Load config.json
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Update model_type
    if config_dict.get("model_type") == "viperlm":
        config_dict["model_type"] = "fireboltlm"
        print("   Updated model_type: viperlm -> fireboltlm")
    
    # Load state dict
    model_files = [
        checkpoint_path / "pytorch_model.bin",
        checkpoint_path / "model.safetensors",
        checkpoint_path / "pytorch_model.bin.index.json",  # For sharded models
    ]
    
    state_dict = None
    for model_file in model_files:
        if model_file.exists():
            if model_file.suffix == ".bin":
                print(f"Loading weights from: {model_file}")
                state_dict = torch.load(model_file, map_location="cpu")
                break
            elif model_file.suffix == ".safetensors":
                try:
                    from safetensors.torch import load_file
                    print(f"Loading weights from: {model_file}")
                    state_dict = load_file(str(model_file))
                    break
                except ImportError:
                    print("Warning: safetensors library not available")
            elif model_file.name == "pytorch_model.bin.index.json":
                # Handle sharded models
                print(f"Found sharded model index: {model_file}")
                with open(model_file, 'r') as f:
                    index = json.load(f)
                state_dict = {}
                for shard_file, weight_map in index.get("weight_map", {}).items():
                    shard_path = checkpoint_path / shard_file
                    if shard_path.exists():
                        shard_dict = torch.load(shard_path, map_location="cpu")
                        # Filter weights for this shard
                        for key in weight_map:
                            if key in shard_dict:
                                state_dict[key] = shard_dict[key]
                break
    
    if state_dict is None:
        raise FileNotFoundError(f"Could not find model weights in {checkpoint_path}")
    
    print(f"Loaded {len(state_dict)} parameters")
    return state_dict, config_dict


def transfer_weights(viper_checkpoint_path: str, output_dir: str):
    """
    Transfer weights from Viper-VL to Firebolt-VL checkpoint.
    """
    print(f"\n{'='*60}")
    print(f"Transferring weights: Viper-VL -> Firebolt-VL")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    print(f"📂 Loading Viper-VL checkpoint from: {viper_checkpoint_path}")
    state_dict, config_dict = load_checkpoint(viper_checkpoint_path)
    
    # Rename state dict keys if needed
    print("\n🔄 Checking and renaming state dict keys...")
    state_dict = rename_state_dict_keys(state_dict)
    
    # Create Firebolt config
    print("\n⚙️  Creating Firebolt-VL configuration...")
    # Remove model_type from config_dict as it's set in the class
    config_dict.pop("model_type", None)
    firebolt_config = FireboltLMConfig(**config_dict)
    firebolt_config.model_type = "fireboltlm"
    
    # Create Firebolt model
    print("🏗️  Creating Firebolt-VL model architecture...")
    try:
        firebolt_model = FireboltLMForCausalLM(firebolt_config)
        print(f"   Model created successfully")
        print(f"   Total parameters: {sum(p.numel() for p in firebolt_model.parameters()):,}")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        raise
    
    # Load state dict
    print("\n🔄 Loading weights...")
    model_state_dict = firebolt_model.state_dict()
    model_keys = set(model_state_dict.keys())
    checkpoint_keys = set(state_dict.keys())
    
    # Find matching keys
    matching_keys = model_keys & checkpoint_keys
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    print(f"   Matching keys: {len(matching_keys)}/{len(model_keys)}")
    
    if missing_keys:
        print(f"   ⚠️  Missing keys: {len(missing_keys)}")
        if len(missing_keys) <= 20:
            for key in sorted(missing_keys):
                print(f"      - {key}")
        else:
            for key in sorted(list(missing_keys)[:10]):
                print(f"      - {key}")
            print(f"      ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"   ⚠️  Unexpected keys (will be ignored): {len(unexpected_keys)}")
        if len(unexpected_keys) <= 20:
            for key in sorted(unexpected_keys):
                print(f"      - {key}")
        else:
            for key in sorted(list(unexpected_keys)[:10]):
                print(f"      - {key}")
            print(f"      ... and {len(unexpected_keys) - 10} more")
    
    # Load weights
    try:
        firebolt_model.load_state_dict(state_dict, strict=False)
        print("   ✅ Weights loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading weights: {e}")
        raise
    
    # Save checkpoint
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving Firebolt-VL checkpoint to: {output_dir}")
    firebolt_model.save_pretrained(str(output_path))
    
    # Copy tokenizer files if they exist
    source_path = Path(viper_checkpoint_path)
    tokenizer_files = [
        "tokenizer_config.json", "vocab.json", "merges.txt", 
        "tokenizer.json", "special_tokens_map.json", "added_tokens.json"
    ]
    
    copied_files = []
    for tokenizer_file in tokenizer_files:
        src_file = source_path / tokenizer_file
        if src_file.exists():
            import shutil
            dst_file = output_path / tokenizer_file
            shutil.copy2(src_file, dst_file)
            copied_files.append(tokenizer_file)
    
    if copied_files:
        print(f"   📋 Copied tokenizer files: {', '.join(copied_files)}")
    
    # Verify saved checkpoint
    saved_config_path = output_path / "config.json"
    if saved_config_path.exists():
        with open(saved_config_path, 'r') as f:
            saved_config = json.load(f)
        if saved_config.get("model_type") == "fireboltlm":
            print(f"   ✅ Config verified: model_type = fireboltlm")
        else:
            print(f"   ⚠️  Warning: model_type in saved config is {saved_config.get('model_type')}")
    
    print(f"\n{'='*60}")
    print(f"✅ Transfer complete!")
    print(f"   Source: {viper_checkpoint_path}")
    print(f"   Output: {output_dir}")
    print(f"   Model type: fireboltlm")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer weights from Viper-VL to Firebolt-VL checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python transfer_weights_simple.py \\
      --viper_checkpoint ./checkpoints/viper-vl/epoch_10 \\
      --output_dir ./checkpoints/firebolt-vl/epoch_10
  
  # With custom device
  CUDA_VISIBLE_DEVICES=0 python transfer_weights_simple.py \\
      --viper_checkpoint ./checkpoints/viper-vl/epoch_10 \\
      --output_dir ./checkpoints/firebolt-vl/epoch_10
        """
    )
    
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
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.viper_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.viper_checkpoint}")
    
    transfer_weights(
        viper_checkpoint_path=args.viper_checkpoint,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()


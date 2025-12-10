# train_ddp_fireboltlm.py
import os
import math
import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from dataset import create_dataloader
from modeling.metric import perplexity
from modeling.model import FireboltLMForCausalLM, FireboltLMConfig
from transformers import AutoProcessor, AutoTokenizer


# -----------------------
# DDP helpers
# -----------------------
def is_distributed(cfg: DictConfig) -> bool:
    return bool(cfg.training.ddp.enabled)

def setup_ddp(cfg: DictConfig):
    if not is_distributed(cfg):
        return None, None, 1
    # Must be launched with torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank: Optional[int]) -> bool:
    return (rank is None) or (rank == 0)


# -----------------------
# Utils
# -----------------------
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(output_dir: str, model, tokenizer, rank: Optional[int]):
    if not is_main_process(rank):
        return
    os.makedirs(output_dir, exist_ok=True)
    to_save = model.module if isinstance(model, DDP) else model
    to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Rank 0] Saved checkpoint to: {output_dir}")

def ddp_all_reduce_mean(value: torch.Tensor, device=None) -> torch.Tensor:
    """All-reduce a tensor across DDP processes. Handles scalars properly."""
    if dist.is_available() and dist.is_initialized():
        # Ensure value is a tensor
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device if device else torch.cuda.current_device())
        
        # Get device from value or use provided device
        if device is None:
            device = value.device if value.is_cuda else torch.cuda.current_device()
        
        # Move to correct device if needed
        if not value.is_cuda:
            value = value.to(device)
        
        # Convert scalar to 1-d tensor for all_reduce
        was_scalar = value.dim() == 0
        if was_scalar:
            value = value.unsqueeze(0)
        
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
        
        # Convert back to scalar if it was originally a scalar
        if was_scalar:
            value = value.squeeze()
    return value


# -----------------------
# Eval
# -----------------------
@torch.no_grad()
def evaluate(rank, model, val_loader, device, amp_dtype) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ppl = 0.0
    n_batches = 0
    n_valid_batches = 0

    iterator = tqdm(val_loader, desc="[Eval]", disable=not is_main_process(rank))
    for batch in iterator:
        if batch is None:
            continue
        try:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)

            with autocast("cuda", dtype=amp_dtype):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_inputs=pixel_values,
                    labels=input_ids,
                    use_cache=False,
                )
                loss = out.loss.detach()

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process(rank):
                    print(f"Warning: NaN/Inf loss detected in evaluation! Skipping this batch. Loss: {loss.item()}")
                continue

            # Perplexity from logits
            ppl = perplexity(out.logits, input_ids).detach()
            
            # Check for NaN/Inf perplexity
            if torch.isnan(ppl) or torch.isinf(ppl):
                if is_main_process(rank):
                    print(f"Warning: NaN/Inf perplexity detected in evaluation! Skipping this batch. PPL: {ppl.item()}")
                continue

            # Average across ranks (only if DDP is enabled)
            if dist.is_available() and dist.is_initialized():
                loss = ddp_all_reduce_mean(loss, device=device)
                ppl = ddp_all_reduce_mean(ppl, device=device)
            
            # Extract scalar values
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            ppl_val = ppl.item() if isinstance(ppl, torch.Tensor) else float(ppl)
            
            # Only accumulate valid values
            if not (math.isnan(loss_val) or math.isinf(loss_val) or math.isnan(ppl_val) or math.isinf(ppl_val)):
                total_loss += loss_val
                total_ppl += ppl_val
                n_valid_batches += 1
            else:
                if is_main_process(rank):
                    print(f"Warning: Invalid loss/ppl values detected: loss={loss_val}, ppl={ppl_val}")
                    
        except Exception as e:
            if is_main_process(rank):
                print(f"Error in evaluation batch: {e}")
                import traceback
                traceback.print_exc()
            continue
        
        n_batches += 1

    if n_valid_batches == 0:
        if is_main_process(rank):
            print("Warning: No valid batches in evaluation!")
        return {"val_loss": float("inf"), "val_ppl": float("inf")}
    
    avg_loss = total_loss / n_valid_batches
    avg_ppl = total_ppl / n_valid_batches
    
    # Final check for NaN/Inf
    if math.isnan(avg_loss) or math.isinf(avg_loss):
        avg_loss = float("inf")
    if math.isnan(avg_ppl) or math.isinf(avg_ppl):
        avg_ppl = float("inf")
    
    return {
        "val_loss": avg_loss,
        "val_ppl": avg_ppl,
    }


# -----------------------
# Train
# -----------------------
def train_one_epoch(cfg, rank, model, train_loader, optimizer, scheduler, device, amp_dtype, scaler):
    model.train()
    # If you freeze the backbone, keep it in eval() to disable dropout etc.
    if getattr(cfg.model, "freeze_llm", False):
        (model.module if isinstance(model, DDP) else model).base_lm.eval()

    running = 0.0
    n_valid_batches = 0
    iterator = tqdm(train_loader,
                    desc=f"[Rank {rank if rank is not None else 0}] Training",
                    disable=not is_main_process(rank))
    cnt = 0
    for batch in iterator:
        # cnt += 1
        # if cnt > 10:
            # break
        if batch is None:
            continue
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=amp_dtype):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_inputs=pixel_values,
                labels=input_ids,
                use_cache=False,
            )
            loss = out.loss
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process(rank):
                    print(f"Warning: NaN/Inf loss detected! Skipping this batch. Loss: {loss.item()}")
                continue

        # AMP (fp16 uses scaler; bf16 skips scaler)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            # IMPORTANT: unscale before clipping
            scaler.unscale_(optimizer)
            # Check for NaN gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.training.get("grad_clip", 1.0),
            )
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if is_main_process(rank):
                    print(f"Warning: NaN/Inf gradient norm detected! Skipping this batch. Grad norm: {grad_norm.item()}")
                scaler.update()  # Update scaler even if skipping step
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.training.get("grad_clip", 1.0),
            )
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if is_main_process(rank):
                    print(f"Warning: NaN/Inf gradient norm detected! Skipping this batch. Grad norm: {grad_norm.item()}")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        scheduler.step()
        loss_item = loss.item()
        if not (math.isnan(loss_item) or math.isinf(loss_item)):
            running += loss_item
            n_valid_batches += 1

        if is_main_process(rank):
            avg_loss = running / max(1, n_valid_batches)
            iterator.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])

    # Return average train loss on rank 0 only (optional)
    avg_train_loss = running / max(1, n_valid_batches)
    
    # Check if loss is valid
    if math.isnan(avg_train_loss) or math.isinf(avg_train_loss):
        if is_main_process(rank):
            print(f"Warning: Average train loss is NaN/Inf: {avg_train_loss}")
        avg_train_loss = float('inf')
    
    return avg_train_loss


# -----------------------
# Main
# -----------------------
@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # Logging (rank 0 only writes file)
    rank, local_rank, world_size = setup_ddp(cfg)
    device = torch.device(f"cuda:{local_rank}" if local_rank is not None else cfg.training.device)
    if is_main_process(rank):
        os.makedirs(cfg.training.results_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cfg.logging.filename), exist_ok=True)
        logging.basicConfig(filename=cfg.logging.filename,
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")

    # Tokenizer/Processor (avoid HF cache race by barrier)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(cfg.processor_path)
    if dist.is_initialized():
        dist.barrier()

    # Model config & load
    vcfg = FireboltLMConfig(**cfg.model)
    # Proper resume: use classmethod .from_pretrained()
    if cfg.training.resume_from_checkpoint:
        if is_main_process(rank):
            print(f"[Rank 0] Resuming from {cfg.training.resume_from_checkpoint}")
        base_model = FireboltLMForCausalLM.from_pretrained(cfg.training.resume_from_checkpoint)
    else:
        base_model = FireboltLMForCausalLM(vcfg)

    # Freeze LLM if requested
    if cfg.model.freeze_llm:
        for p in base_model.base_lm.parameters():
            p.requires_grad = False
        base_model.base_lm.eval()
    else:
        base_model.base_lm.train()
        for p in base_model.base_lm.parameters():
            p.requires_grad = True

    if cfg.model.get("vision_freeze", True):
        for p in base_model.model.vision_encoder.parameters():
            p.requires_grad = False
        base_model.model.eval()
    else:
        base_model.model.train()
        for p in base_model.model.vision_encoder.parameters():
            p.requires_grad = True
    
    if cfg.model.get("fuser_freeze", True):
        for p in base_model.model.fuser.parameters():
            p.requires_grad = False
        base_model.model.eval()
    else:
        base_model.model.fuser.train()
        for p in base_model.model.fuser.parameters():
            p.requires_grad = True
    # Device & dtype
    use_bf16 = (cfg.training.amp_dtype == "bf16")
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    base_model.to(device)
    base_model.train()
    # DDP wrap (only if enabled)
    if is_distributed(cfg):
        base_model = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=cfg.training.ddp.get("find_unused_parameters", True),
        )

    # Image token id (optional bookkeeping)
    (base_model.module if isinstance(base_model, DDP) else base_model).config.image_token_id = \
        tokenizer.convert_tokens_to_ids("<image>")

    # DataLoaders (ensure ddp flag + rank/world_size passed through)
    loaders = create_dataloader(
        image_path=cfg.data.image_path,
        json_path=cfg.data.json_path,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        ddp=is_distributed(cfg),
        rank=(rank if rank is not None else 0),
        world_size=world_size,
        train_val_split=cfg.data.train_val_split,
    )
    train_loader = loaders["train_dataloader"]
    val_loader = loaders["val_dataloader"]
    train_sampler = loaders.get("train_sampler")
    if dist.is_initialized():
        dist.barrier()

    # Optimizer / Scheduler / Scaler
    trainable = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg.training.optimizer.lr,
        betas=tuple(cfg.training.optimizer.get("betas", (0.9, 0.95))),
        eps=cfg.training.optimizer.get("eps", 1e-8),
        weight_decay=cfg.training.optimizer.get("weight_decay", 0.01),
    )
    total_steps = cfg.training.num_epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=cfg.training.scheduler.eta_min
    )
    scaler = GradScaler("cuda", enabled=not use_bf16)

    if is_main_process(rank):
        print(f"Trainable parameters: {count_trainable_parameters(base_model):,}")

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        avg_train = train_one_epoch(
            cfg, rank, base_model, train_loader, optimizer, scheduler, device, amp_dtype, scaler
        )
        metrics = evaluate(rank, base_model, val_loader, device, amp_dtype)

        if is_main_process(rank):
            print(f"[Epoch {epoch+1}] "
                  f"train_loss={avg_train:.4f}  val_loss={metrics['val_loss']:.4f}  val_ppl={metrics['val_ppl']:.2f}")
            logging.info(f"[Epoch {epoch+1}] "
                         f"train_loss={avg_train:.4f}  val_loss={metrics['val_loss']:.4f}  val_ppl={metrics['val_ppl']:.2f}")
            save_checkpoint(os.path.join(cfg.training.results_dir, f"epoch_{epoch+1}"),
                            base_model, tokenizer, rank)

    cleanup_ddp()


if __name__ == "__main__":
    # When using DDP, launch with:
    #   torchrun --nproc_per_node=NUM_GPUS train_ddp_viperlm_fixed.py
    main()

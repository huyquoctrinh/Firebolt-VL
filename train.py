import hydra
import torch
import logging
from tqdm import tqdm
from omegaconf import DictConfig
from dataset import create_dataloader
from modeling.metric import perplexity
from modeling.model import FireboltLMForCausalLM, FireboltLMConfig
from transformers import AutoProcessor, AutoTokenizer
from time import time
import os
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def count_trainable_parameters(model):
    """
    Counts the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval_model(model, val_dataloader, device):
    """
    Evaluates the model on the validation set, calculating loss and perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_perplexity = 0.0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating", unit="batch"):
            if batch is None: continue
            input_ids = batch["input_ids"].to(device)
            images = batch["pixel_values"].to(device)
            attention_mask = batch.get("attention_mask").to(device)

            outputs = model(
                input_ids=input_ids, 
                image_inputs=images, 
                labels=input_ids, 
                attention_mask=attention_mask
            )
            
            total_loss += outputs.loss.item()
            perplexity_value = perplexity(outputs.logits, input_ids)
            total_perplexity += perplexity_value.item()

    avg_loss = total_loss / len(val_dataloader)
    avg_perplexity = total_perplexity / len(val_dataloader)
    return avg_loss, avg_perplexity

def train(
    cfg: DictConfig,
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device
):
    """
    Main training loop.
    """
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}", unit="batch")
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if cfg.training.amp_dtype == "bf16" else torch.float16):
            for batch in progress_bar:
                if batch is None: continue
                input_ids = batch["input_ids"].to(device)
                images = batch["pixel_values"].to(device)
                attention_mask = batch.get("attention_mask").to(device)
                
                optimizer.zero_grad()
                
                outputs = model(
                    input_ids=input_ids, 
                    image_inputs=images, 
                    labels=input_ids,
                    attention_mask=attention_mask
                )
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                progress_bar.set_postfix(loss=total_train_loss / (progress_bar.n + 1))

            scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Evaluation
        avg_val_loss, avg_perplexity = eval_model(model, val_dataloader, device)
        
        print(f"Epoch [{epoch+1}/{cfg.training.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Perplexity: {avg_perplexity:.4f}")
        logging.info(f"Epoch [{epoch+1}/{cfg.training.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Perplexity: {avg_perplexity:.4f}")
        print("======================================================")
        
        model.save_pretrained(f"{cfg.training.results_dir}/epoch_{epoch+1}")

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # --- Setup ---
    logging.getLogger().handlers.clear()
    os.makedirs(cfg.training.results_dir, exist_ok=True)
    os.makedirs("./" + cfg.logging.filename.split("/")[0], exist_ok=True)
    logging.basicConfig(filename="./" + cfg.logging.filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = cfg.training.device

    # --- Model ---
    config = FireboltLMConfig(**cfg.model)
    model = FireboltLMForCausalLM(config)

    if cfg.training.resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {cfg.training.resume_from_checkpoint}")
        model.from_pretrained(cfg.training.resume_from_checkpoint)
    model.to(device)
    
    if config.freeze_llm:
        for p in model.base_lm.parameters():
            p.requires_grad = False
    else:
        model.base_lm.resize_token_embeddings(config.vocab_size)
        model.base_lm.train()
        for p in model.base_lm.parameters():
            p.requires_grad = True

    # --- Tokenizer & Processor ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    processor = AutoProcessor.from_pretrained(cfg.processor_path)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model.config.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    # --- DataLoaders ---
    dataloaders = create_dataloader(
        image_path=cfg.data.image_path,
        json_path=cfg.data.json_path,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        ddp=cfg.training.ddp.enabled,
        rank=cfg.training.ddp.rank,
        world_size=cfg.training.ddp.world_size,
        train_val_split=cfg.data.train_val_split
    )

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.optimizer.lr)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.scheduler.T_max, eta_min=cfg.training.scheduler.eta_min
    )

    print(f"Number of trainable parameters: {count_trainable_parameters(model)}")

    # --- Train ---
    train(cfg, model, dataloaders["train_dataloader"], dataloaders["val_dataloader"], optimizer, cosine_scheduler, device)

if __name__ == "__main__":
    main()

# evaluate.py
import torch
from PIL import Image
from transformers import AutoTokenizer, GenerationConfig, AutoProcessor
import hydra
from omegaconf import DictConfig
from typing import Optional

# --- Project Modules ---
from modeling.model import FireboltLMForCausalLM

@torch.inference_mode()
def run_inference(
    cfg: DictConfig,
    model: FireboltLMForCausalLM,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    prompt: str,
    image_path: Optional[str] = None,
):
    """
    Runs a single inference example using the model's chat template.
    """
    device = cfg.training.device
    
    # --- Prepare Inputs using Chat Template ---
    image_inputs = None
    if image_path:
        try:
            # Prepare image
            img = Image.open(image_path).convert("RGB")
            image_inputs = processor(images=[img], return_tensors="pt")["pixel_values"].to(device, dtype=model.dtype)
            
            # Format prompt with image token
            user_content = f"<image>\n{prompt}"
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Running text-only inference.")
            user_content = prompt
    else:
        user_content = prompt

    messages = [{"role": "user", "content": user_content}]
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)

    # --- Generate ---
    
    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.evaluation.max_new_tokens,
        do_sample=cfg.evaluation.do_sample,
        temperature=cfg.evaluation.get("temperature", 0.7),
        top_p=cfg.evaluation.get("top_p", 0.9),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

    gen_kwargs = {}
    if image_inputs is not None:
        # ensure dtype/device match model
        model_dtype = next(model.parameters()).dtype
        image_inputs = image_inputs.to(device, dtype=model_dtype)
        gen_kwargs["image_inputs"] = image_inputs
        print("Image inputs: ", image_inputs.shape)

    print("Generating response...")
    with torch.autocast(device_type=device, dtype=model.dtype):
        gen_ids = model.generate(inputs, generation_config=gen_cfg, **gen_kwargs)
    
    # Decode only the newly generated tokens
    prompt_len = inputs.shape[1]
    new_tokens = gen_ids[0, prompt_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print("\n--- Prompt ---")
    print(prompt)
    if image_path:
        print(f"(with image: {image_path})")
    print("\n--- Response ---")
    # print(decoded)
    print("### Reasoning Steps ###")
    print("\n".join(decoded.split("\n")[:-1]).strip())
    print("### Final Answer ###")
    print(decoded.split("\n")[-1].strip())

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # --- Device and Dtype ---
    device = cfg.training.device
    torch_dtype = getattr(torch, cfg.training.amp_dtype, torch.float16)

    # --- Load Model, Tokenizer, and Processor ---
    print(f"Loading model from {cfg.evaluation.model_dir}...")
    model = FireboltLMForCausalLM.from_pretrained(
        cfg.evaluation.model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    tokenizer_dir = cfg.evaluation.tokenizer_dir or cfg.evaluation.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(cfg.processor_path)
    model.config.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    print("Successfully loaded model, tokenizer, and processor.")
    # --- Run Inference ---
    run_inference(
        cfg,
        model,
        tokenizer,
        processor,
        prompt=cfg.evaluation.prompt,
        image_path=cfg.evaluation.image_path,
    )

if __name__ == "__main__":
    main()

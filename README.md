# 🔥 **FIREBOLT-VL: A Family of Small Multimodal-LLMs**

<div align="center">
  <a href="./">
    <img src="assets/firebolt-vl_represent.png" width="80%" alt="Firebolt-VL Logo"/>
  </a>
  <br/>
  <i>"Fast. Compact. Vision-Language Intelligence."</i>
</div>

***Note:*** This model is still in improving, so we recommend to fine-tune this model in your use case!

---

## 🌟 Overview

**Firebolt-VL** is an open-source **small multimodal large language model (Multimodal-LLM)** designed for efficient multimodal reasoning and deployment on consumer GPUs.
It is built upon the [**Liquid Model**](https://huggingface.co/LiquidAI/LFM2-350M) architecture (≈1.2B parameters), enabling a powerful yet lightweight foundation for **personal research, on-device applications, and internal experimentation**.

---

## 🧠 Key Features

* ⚡ **Efficient Training & Inference**
  Trained on **2× H100 GPUs** within **~2 days**, thanks to our lightweight multimodal fusion and liquid transformer design.
  Inference runs smoothly even on **RTX 4070** GPUs.

* 🔗 **Multimodal Connector (Sense Integration Module)**
  Inspired by human perception, Firebolt-VL introduces a *connector* that fuses signals from different sensory encoders (vision, audio, etc.), enabling deeper **cross-modal alignment** and improved reasoning.

* 🧩 **Hybrid Architecture**
  Combines the **semantic strength of Transformers** with the **efficiency of Liquid Neural Networks**, resulting in a compact yet expressive multimodal model.

---

## 🚀 Progress

* ✅ **Released** — Firebolt-VL model checkpoint
* ✅ **Released** — Firebolt-VL inference code
* ✅ **Released** — Training script
* 🧩 **Coming Soon** — Fully documented training and inference scripts
* 🧩 **Coming Soon** — Fully documented for post-training (LoRA, DPO, GRPO)

Stay tuned for our next updates on model fine-tuning and multimodal reasoning enhancements.

---

## 🏗️ Architecture

The overall architecture is shown below:

<div align="center">
  <a href="./">
    <img src="assets/firebolt-vl.png" width="80%" alt="Firebolt-VL Architecture"/>
  </a>
</div>

**Main Components:**

1. 🎨 **Vision Encoder** – Extracts compact visual embeddings
2. 🔗 **Multimodal Connector** – Fuses sensory inputs efficiently
3. 🧠 **Language Backbone (LFM2-350M-based)** – Performs semantic reasoning and response generation

> 🧪 *The current Firebolt-VL (1.2B parameters) was trained on ~4 million images using 2× H100 GPUs for 2 days.*

---

## 📊 Benchmark Results

**Number of parameters:** 740M

| Benchmark   | Task | Split | Metric   | Firebolt-VL (CoT) |
|-------------|------|-------|----------|----------------|
| RealWorldQA | VQA  | Test  | Accuracy | **38.30%**     |
| VQA-V2 | VQA  | Test  | Accuracy | **53.18%**   |

**Notes.** CoT = Chain-of-Thought prompting enabled during inference. Exact settings (temperature/top-p/max tokens) can influence results; see the inference snippet below to replicate typical generation settings.

---

## 🧩 Usage

To get started with **inference**, follow the setup in the main repository:

🔗 [**Firebolt-VL Repository**](https://github.com/huyquoctrinh/Firebolt-LM)
📜 Example inference script: [`infer.sh`](https://github.com/huyquoctrinh/Firebolt-LM/blob/feat/main/infer.sh)

Or you can use these functions for inference

```python
import os
import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from modeling.model import FireboltLMForCausalLM  # your local class
IMAGE_TOKEN_ID = 64400
def build_messages(question: str, include_image: bool = True):
    # Mirror CCDataset._format_prompt()
    user_content = ("<image> " if include_image else "") + (question or "")
    return [
        {"role": "user", "content": user_content},
        # assistant turn is left empty; apply_chat_template(add_generation_prompt=True) will add assistant prefix
    ]

@torch.inference_mode()
def generate_answer(
    ckpt_dir: str,
    tokenizer_path: str,
    processor_path: str,
    image_path: str,
    question: str,
    device: str = "cuda",
    dtype: str = "bf16",
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
):
    # --- device / dtype ---
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    use_bf16 = (dtype.lower() == "bf16")
    use_fp16 = (dtype.lower() == "fp16")
    amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # --- tokenizer / processor ---
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # optional but common for generation with left context
    if not hasattr(tokenizer, "padding_side") or tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    processor = AutoProcessor.from_pretrained(processor_path)

    # --- model ---
    model = FireboltLMForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=amp_dtype if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # expose image token id if your forward expects it; keep it consistent with training
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None and "<image>" in tokenizer.get_vocab():
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    # --- text input with the SAME chat template as training ---
    messages = build_messages(question=question, include_image=True)
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,   # adds assistant header the model expects before generation
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(enc, torch.Tensor):
        input_ids = enc
        attention_mask = torch.ones_like(enc, dtype=torch.long)
    else:
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # --- image preprocessing (match training) ---
    img = Image.open(image_path).convert("RGB")
    proc = processor(images=[img], return_tensors="pt")  # list, like training
    pixel_values = proc.get("pixel_values", None)
    if pixel_values is None:
        raise ValueError("Processor did not return 'pixel_values'. Check processor_path.")
    pixel_values = pixel_values.to(device)  # (1, 3, H, W)

    # --- generate ---
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": max(temperature, 1e-6),
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "image_inputs": pixel_values,
        # IMPORTANT: use the same argument names your model.forward saw in training           # not "image_inputs"
        "image_token_id": image_token_id,       # if your forward uses it
        "use_cache": False,
    }

    if device.type == "cuda" and (use_bf16 or use_fp16):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
    else:
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )

    # --- decode only new tokens ---
    generated = out[0]
    prompt_len = input_ids.size(1)
    new_tokens = generated[prompt_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer.strip()

if __name__ == "__main__":
    ckpt_dir = ""
    tokenizer_path = ""
    processor_path = ""
    image_path = ""
    question = ""
    device = ""
    ans = generate_answer(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        processor_path=processor_path,
        image_path=image_path,
        question=question,
        device=device,
        dtype="bfloat16",
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1
    )
    print("\n ======Answer===== \n")
    print(ans)

```

---

## 🙏 Acknowledgements

We gratefully thank the following foundational projects for inspiring and enabling our research:

* [**Liquid Model**](https://huggingface.co/LiquidAI/LFM2-350M) – Base architecture for dynamic neural computation
* [**SigLIP**](https://huggingface.co/google/siglip2-base-patch16-naflex) – Vision encoder powering multimodal understanding

Their open-source contributions have made **Firebolt-VL** possible. 💚

---

## 📫 Contact

If you’re interested in collaboration or research discussions:
👉 [**Contact us**](https://github.com/huyquoctrinh) or open an issue in the repository.

---
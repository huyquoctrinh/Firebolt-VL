# 🔥 **FIREBOLT-VL: Efficient Vision-Language Understanding with Cross-Modality Modulation**

***Note:*** Firebolt-VL is under active development. For best results on your domain, we recommend fine-tuning on your target data.

---

## 🌟 Overview

**Firebolt-VL** is an open-source **efficient vision-language model (VLM)** designed for fast multimodal reasoning and practical deployment.
It uses the [**Liquid Foundation Model (LFM2-350M)**](https://huggingface.co/LiquidAI/LFM2-350M) as the language decoder and introduces a lightweight **Cross-modal Modulator (CMM)** to improve fine-grained visual grounding while keeping inference efficient.

Compared to Transformer-style cross-attention fusion, Firebolt-VL emphasizes **token–grid correlation + FiLM conditioning + state-space sequence modeling** for near-linear scaling with sequence length.

---

## 🧠 Key Features

* ⚡ **Efficient inference**
  Replaces heavy cross-attention fusion with a lightweight CMM and state-space modeling for fast generation.

* 🎯 **Fine-grained visual grounding**
  Computes **token–grid correlations** and uses **Top-K grid selection** so each text token focuses on the most relevant visual regions.

* 🧩 **Cross-modality modulation (FiLM)**
  Uses **Feature-wise Linear Modulation (FiLM)** to inject image-conditioned signals into text features without quadratic attention overhead.

---

## 🚀 Project Status

* ✅ **Released** — Model checkpoint(s)
* ✅ **Released** — Inference code
* ✅ **Released** — Training script(s)
* 🧩 **Coming Soon** — More documentation (training recipes, evaluation, post-training: LoRA/DPO/GRPO)

---

## 🏗️ Architecture

<div align="center">
  <a href="./">
    <img src="assets/firebolt_vl.jpg" width="85%" alt="Firebolt-VL Architecture"/>
  </a>
</div>

**Main Components:**
1. 🎨 **Vision Encoder (SigLIP)** – extracts grid-level visual embeddings  
2. 🧩 **Cross-modal Modulator (CMM)** – token–grid correlation → FiLM → SSM → FiLM  
3. 🧠 **Language Decoder (LFM2-350M-based)** – multimodal reasoning and response generation  

> Training in the paper follows a **two-stage recipe**: (1) CMM warm-up with frozen backbones, then (2) end-to-end training.

---

## 📊 Benchmark Results

**Total parameters (paper setting):** ~0.8B

| Benchmark | Split | Score |
|---|---:|---:|
| VQAv2 | Test | **76.6** |
| POPE | Test | **69.4** |
| AI2D | Test | **46.2** |
| MMMU-val | Val | **26.4** |
| MME (Perception) | - | **1376.2** |
| SQA-Image | Test | **56.7** |
| MMB-dev | Dev | **64.6** |

**Notes.** Results can vary with decoding parameters and evaluation pipelines.

---

## 🧩 Usage

To get started with inference, follow the setup in the repository:

🔗 **Firebolt-VL Repository:** https://github.com/huyquoctrinh/Firebolt-VL  
📜 Example inference script: `infer.sh`

### Minimal inference example (template)

> Update the model import / forward kwargs to match your implementation.

```python
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from modeling.model import FireboltLMForCausalLM  # local class

def build_messages(question: str, include_image: bool = True):
    user_content = ("<image> " if include_image else "") + (question or "")
    return [{"role": "user", "content": user_content}]

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
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if dtype.lower() in ["bf16", "bfloat16"] else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    processor = AutoProcessor.from_pretrained(processor_path)

    model = FireboltLMForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=amp_dtype if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    messages = build_messages(question=question, include_image=True)
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    input_ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    img = Image.open(image_path).convert("RGB")
    proc = processor(images=[img], return_tensors="pt")
    pixel_values = proc["pixel_values"].to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        # NOTE: rename this kwarg if your model expects a different name
        image_inputs=pixel_values,
    )

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    else:
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    generated = out[0]
    new_tokens = generated[input_ids.size(1):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

if __name__ == "__main__":
    print(generate_answer(
        ckpt_dir="YOUR_CKPT",
        tokenizer_path="YOUR_TOKENIZER",
        processor_path="YOUR_PROCESSOR",
        image_path="demo.jpg",
        question="What is the destination city searched in this image?"
    ))
````

---

## 🙏 Acknowledgements

We gratefully thank the following foundational projects:

* [**Liquid Foundation Model (LFM2-350M)**](https://huggingface.co/LiquidAI/LFM2-350M) – language decoder backbone
* [**SigLIP / SigLIP2**](https://huggingface.co/google/siglip2-base-patch16-naflex) – vision encoder backbone

---

## 📫 Contact

For questions, collaborations, or issues:
👉 [https://github.com/huyquoctrinh](https://github.com/huyquoctrinh) (open an issue or reach out via GitHub)

---


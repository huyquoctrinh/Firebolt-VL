
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForCausalLM

from .vision_encoder.siglip import VisionEncoder
from .vision_encoder.siglip_grid import GridVisionEncoder
from .factory import build_projector, build_fuser

class FireboltLMConfig(PretrainedConfig):
    model_type = "fireboltlm"

    def __init__(
        self,
        lm_name_or_path: str = "LiquidAI/LFM2-350M",
        image_token_id: Optional[int] = None,
        vision_hidden_size: int = 768,
        vision_output_tokens: int = 256,
        vision_freeze: bool = True,
        vision_ckpt_path: Optional[str] = None,
        projector_type: str = "residual_ffn",
        projector_hidden: int = 768,
        projector_layers: int = 2,
        projector_dropout: float = 0.1,
        fuser_type: str = "multimodal_fusing",
        fuse_strategy: str = "prepend",
        fuse_heads: int = 8,
        fuse_attn_dropout: float = 0.1,
        fuse_proj_dropout: float = 0.0,
        amp_dtype: str = "fp16",
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.lm_name_or_path = lm_name_or_path
        self.image_token_id = image_token_id
        self.vision_hidden_size = vision_hidden_size
        self.vision_output_tokens = vision_output_tokens
        self.vision_freeze = vision_freeze
        self.vision_ckpt_path = vision_ckpt_path
        self.projector_type = projector_type
        self.projector_hidden = projector_hidden
        self.projector_layers = projector_layers
        self.projector_dropout = projector_dropout
        self.fuser_type = fuser_type
        self.fuse_strategy = fuse_strategy
        self.fuse_heads = fuse_heads
        self.fuse_attn_dropout = fuse_attn_dropout
        self.fuse_proj_dropout = fuse_proj_dropout
        self.amp_dtype = amp_dtype
        self.use_cache = use_cache

class FireboltLMModel(nn.Module):
    def __init__(self, config: FireboltLMConfig, lm_hidden_size: int):
        super().__init__()
        self.config = config

        cur_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        cur_dtype = torch.bfloat16 if getattr(config, "amp_dtype", "fp16") == "bf16" else torch.float16
        # self.config.num_hidden_layers = config.n_layers

        if not config.vision_ckpt_path:
            print("Warning: `vision_ckpt_path` not set. Vision encoder will not be initialized.")
            self.vision_encoder = None
        elif config.fuse_strategy != "cossm":
            self.vision_encoder = VisionEncoder(
                ckpt_path=config.vision_ckpt_path,
                device=cur_device,
                dtype=cur_dtype,
            )
            if config.vision_freeze:
                for p in self.vision_encoder.parameters():
                    p.requires_grad = False
            else:
                self.vision_encoder.train()
                for p in self.vision_encoder.parameters():
                    p.requires_grad = True
        elif config.fuse_strategy == "cossm":
            self.vision_encoder = GridVisionEncoder(
                ckpt_path=config.vision_ckpt_path
            )
            if config.vision_freeze:
                for p in self.vision_encoder.parameters():
                    p.requires_grad = False
            else:
                self.vision_encoder.train()
                for p in self.vision_encoder.parameters():
                    p.requires_grad = True

        self.projector = build_projector(
            config.projector_type,
            input_dim=config.vision_hidden_size,
            output_dim=lm_hidden_size,
            hidden_dim=config.projector_hidden,
            dropout=config.projector_dropout,
        )
        self.fuser = build_fuser(
            config.fuser_type,
            llm_dim=lm_hidden_size,
            strategy=config.fuse_strategy,
            num_heads=config.fuse_heads,
            attn_dropout=config.fuse_attn_dropout,
            proj_dropout=config.fuse_proj_dropout,
        )
        self.embed_dropout = nn.Dropout(0.0)

    def _encode_images(self, image_inputs: torch.Tensor) -> torch.Tensor:
        if self.vision_encoder is None:
            raise ValueError("Vision encoder is not initialized. Cannot process images.")
        with torch.set_grad_enabled(not self.config.vision_freeze):
            vis_tokens = self.vision_encoder(image_inputs)
        return self.projector(vis_tokens)
    
    def _get_vision_tokens(self, image_inputs: torch.Tensor) -> torch.Tensor:
        """Get raw vision tokens before projection (for cossm/film strategies)."""
        if self.vision_encoder is None:
            raise ValueError("Vision encoder is not initialized. Cannot process images.")
        with torch.set_grad_enabled(not self.config.vision_freeze):
            # For cossm/film, we need patch embeddings, not pooled features
            # Check if vision encoder is GridVisionEncoder (returns patch-like tokens) or regular VisionEncoder
            if hasattr(self.vision_encoder, 'encoder'):
                # GridVisionEncoder - already returns (B, num_tiles, D)
                vis_tokens = self.vision_encoder(image_inputs)
            else:
                # Regular VisionEncoder - need to get patch embeddings from vision_model
                if hasattr(self.vision_encoder, 'model') and hasattr(self.vision_encoder.model, 'vision_model'):
                    # Get device from model
                    device = next(self.vision_encoder.model.parameters()).device
                    image_inputs = image_inputs.to(device)
                    outputs = self.vision_encoder.model.vision_model(image_inputs)
                    vis_tokens = outputs.last_hidden_state  # (B, num_patches+1, 768)
                else:
                    # Fallback: use pooled features and add sequence dimension
                    vis_tokens = self.vision_encoder(image_inputs)  # (B, D)
                    if vis_tokens.dim() == 2:
                        vis_tokens = vis_tokens.unsqueeze(1)  # (B, 1, D)
        return vis_tokens
    
    def process_multimodal_input_ids(
        self,
        batch_input_ids,
        batch_input_ids_embedding,
        image_embeddings,
        image_token_id: int,
    ):
        multimodal_embeddings = []
        for i, batch in enumerate(batch_input_ids):
            multimodal_embedding = []
            for j, token_id in enumerate(batch):
                if token_id == image_token_id:
                    # print("Image embedding shape:", image_embeddings[i].shape)
                    multimodal_embedding.append(image_embeddings[i])
                else:
                    # print("Token embedding shape:", batch_input_ids_embedding[i][j].shape)
                    multimodal_embedding.append(batch_input_ids_embedding[i][j])

            multimodal_embeddings.append(torch.stack(multimodal_embedding))
        return torch.stack(multimodal_embeddings, dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embedding_layer: nn.Embedding,
        attention_mask: Optional[torch.Tensor] = None,
        image_inputs: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
    ):
        text_embeds = self.embed_dropout(input_embedding_layer(input_ids))

        if image_inputs is not None and vision_embeds is None:
            # For cossm/film strategies, fuser needs raw vision tokens (768-dim)
            # For other strategies, use projected embeddings (1024-dim)
            if self.config.fuse_strategy in ["cossm", "film"]:
                vision_embeds = self._get_vision_tokens(image_inputs)  # Raw vision tokens (B, G, 768)
            else:
                vision_embeds = self._encode_images(image_inputs)  # Projected embeddings (B, G, 1024)
            # print("Vision embeddings: ", vision_embeds.shape)
        if vision_embeds is not None:
            fused_embeds = self.fuser(text_embeds, vision_embeds)
            # For cossm/film, fused_embeds is already (B, T, D) - per-token fused embeddings
            # For other strategies, we need to process with image_token_id
            if self.config.fuse_strategy not in ["cossm", "film"]:
                fused_embeds = self.process_multimodal_input_ids(
                    input_ids,
                    text_embeds,
                    fused_embeds,
                    image_token_id,
                )
            # print("Fused embeddings: ", fused_embeds.shape)
            if self.config.fuse_strategy == 'prepend' and attention_mask is not None:
                vision_mask = torch.ones(vision_embeds.shape[:2], dtype=torch.long, device=vision_embeds.device)
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
            return fused_embeds, attention_mask
        
        return text_embeds, attention_mask

class FireboltLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = FireboltLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    main_input_name = "input_ids"

    def __init__(self, config: FireboltLMConfig):
        super().__init__(config)
        self.base_lm = AutoModelForCausalLM.from_pretrained(config.lm_name_or_path)
        lm_hidden = self.base_lm.get_input_embeddings().embedding_dim
        self.config.vocab_size = self.base_lm.config.vocab_size
        self.model = FireboltLMModel(config, lm_hidden_size=lm_hidden)
        self.post_init()
        
    def get_input_embeddings(self) -> nn.Embedding:
        return self.base_lm.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.base_lm.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.base_lm.get_output_embeddings()

    def set_output_embeddings(self, new_lm_head: nn.Module) -> None:
        if hasattr(self.base_lm, "set_output_embeddings"):
            self.base_lm.set_output_embeddings(new_lm_head)

    def tie_weights(self) -> None:
        if hasattr(self.base_lm, "tie_weights"):
            self.base_lm.tie_weights()

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "image_inputs": kwargs.get("image_inputs", None),
            "vision_embeds": kwargs.get("vision_embeds", None),
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_inputs: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_token_id: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:        
        if inputs_embeds is None:
            inputs_embeds, attention_mask = self.model(
                input_ids=input_ids,
                input_embedding_layer=self.get_input_embeddings(),
                attention_mask=attention_mask,
                image_inputs=image_inputs,
                vision_embeds=vision_embeds
            )

        return self.base_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        emb = self.base_lm.resize_token_embeddings(new_num_tokens)
        self.config.vocab_size = new_num_tokens
        return emb

    def _set_gradient_checkpointing(self, module, value: bool = False):
        if hasattr(self.base_lm, "_set_gradient_checkpointing"):
            self.base_lm._set_gradient_checkpointing(module, value)

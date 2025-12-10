
import torch
import torch.nn as nn
from typing import Optional, Dict

from .base import Fuser
from .factory import register_fuser
from .cossm import TokenGridCorrFiLMS4, GridS4Fuser
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=16, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        output = query + attn_output
        return output

@register_fuser("multimodal_fusing")
class MultimodalFusing(Fuser):
    """
    Fuse text embeddings (B, T, H) with vision embeddings (B, Tv, H).
    """
    def __init__(
        self,
        llm_dim: int,
        strategy: str = "prepend",
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.0,
        vision_dim: Optional[int] = 768,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.strategy = strategy

        if strategy == "cross_attend":
            self.kv_proj = nn.Linear(llm_dim, llm_dim, bias=False)
            self.q_proj  = nn.Linear(llm_dim, llm_dim, bias=False)
            self.attn = nn.MultiheadAttention(
                embed_dim=llm_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            self.out_proj = nn.Sequential(
                nn.Linear(llm_dim, llm_dim, bias=False),
                nn.Dropout(proj_dropout),
            )
            self.norm = nn.LayerNorm(llm_dim)
        elif strategy == "cossm":
            self.fuser = GridS4Fuser(
                d_text=llm_dim,
                d_vision_in=vision_dim,
                heads=4,
                head_dim=32,
                drop=0.1,
                ffn_mult=4.0,
                proj_vision_to_text=True,
                topk=4,
                ssm_type="s4",
            )

        elif strategy == "film":
            self.fuser = GridS4Fuser(
                d_text=llm_dim,
                d_vision_in=vision_dim,
                heads=4,
                head_dim=32,
                drop=0.1,
                ffn_mult=4.0,
                proj_vision_to_text=True,
                topk=1,
                ssm_type="s4",
            )
        else:
            raise NotImplementedError(f"Fusing strategy '{strategy}' not implemented.")

    def forward(
        self,
        text_embeds: torch.Tensor,
        vision_embeds: Optional[torch.Tensor], 
        extra: Optional[Dict] = None,
    ) -> torch.Tensor:
        if vision_embeds is None:
            return text_embeds

        if self.strategy == "prepend":
            # return torch.cat([vision_embeds, text_embeds], dim=1)
            return text_embeds
        # ---- cross_attend ----
        elif self.strategy == "cross_attend":
            vision_embeds = vision_embeds.unsqueeze(1)  # (B, 1, H)
            q = self.q_proj(vision_embeds)
            kv = self.kv_proj(text_embeds)
            attn_out, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
            fused = text_embeds + self.out_proj(attn_out)
            fused = self.norm(fused)
            return fused.squeeze(1)
        elif self.strategy == "cossm":
            # print("Using COSM fuser")
            # print("Text embeddings: ", text_embeds.shape)
            # print("Vision embeddings: ", vision_embeds.shape)
            text_embeds = self.fuser(
                x_text=text_embeds,
                x_vision=vision_embeds,
            )
        elif self.strategy == "film":
            # Film strategy uses the same GridS4Fuser as cossm
            text_embeds = self.fuser(
                x_text=text_embeds,
                x_vision=vision_embeds,
            )

        return text_embeds

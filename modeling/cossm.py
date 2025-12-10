import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssm.s4 import S4Block
from .ssm.s4d import S4D

class TokenGridCorrFiLMS4(nn.Module):
    """
    Token↔Grid correlation -> per-token context -> FiLM -> S4 over text (no attention op).
    Shapes:
      x_text:   (B, T, D_text)
      x_vision: (B, G, 768)
      text_mask:   optional (B, T)  True=keep, False=pad
      vision_mask: optional (B, G)  True=keep, False=pad
    """
    def __init__(
        self,
        d_text: int,
        s4_ctor,                 # callable: s4_ctor(d_model) -> nn.Module mapping (B,T,D)->(B,T,D)
        d_vision_in: int = 768,  # your grid embedding dim
        heads: int = 4,
        head_dim: int = 32,
        drop: float = 0.1,
        ffn_mult: float = 4.0,
        proj_vision_to_text: bool = True,  # set False if your vision is already in D_text
        topk: int = 0,  # 0 = full softmax over grids; >0 = keep top-k grids per token (sparser/faster)
    ):
        super().__init__()
        self.d_text = d_text
        self.heads = heads
        self.head_dim = head_dim
        self.inner = heads * head_dim
        self.topk = topk

        # Project vision to a space aligned to text for correlation + context
        self.k_proj = nn.Linear(d_vision_in, self.inner, bias=False)           # for correlation
        self.v_proj = nn.Linear(d_vision_in, d_text if proj_vision_to_text else d_vision_in, bias=False)  # for context
        # If we kept v in original space, add a projection to d_text before FiLM:
        self.ctx_to_text = None
        if not proj_vision_to_text:
            self.ctx_to_text = nn.Linear(d_vision_in, d_text, bias=False)

        # Project text to Q for correlation
        self.q_proj = nn.Linear(d_text, self.inner, bias=False)
        self.scale = 1.0 / math.sqrt(head_dim)

        self.drop_attn = nn.Dropout(drop)

        # FiLM (per-token) around S4
        self.film_in  = nn.Linear(d_text, 2 * d_text)   # scale, shift
        self.film_out = nn.Linear(d_text, 2 * d_text)

        # S4 backbone
        self.norm_in  = nn.LayerNorm(d_text)
        self.s4       = s4_ctor(d_text)
        self.norm_out = nn.LayerNorm(d_text)

        # FFN head
        hidden = int(ffn_mult * d_text)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_text),
            nn.Linear(d_text, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_text),
            nn.Dropout(drop),
        )

        # small gate so modulation can ramp up safely
        self.mod_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_text: torch.Tensor, x_vision: torch.Tensor,
                text_mask: torch.Tensor = None, vision_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x_text:   (B, T, D_text)
        x_vision: (B, G, 768)
        """
        B, T, D = x_text.shape
        G = x_vision.size(1)

        # ---- 1) Multi-head token↔grid correlation (softmax over grids) ----
        Q = self.q_proj(x_text)            # (B, T, H*Dh)
        K = self.k_proj(x_vision)          # (B, G, H*Dh)

        H, Dh = self.heads, self.head_dim
        Q = Q.view(B, T, H, Dh).transpose(1, 2)   # (B, H, T, Dh)
        K = K.view(B, G, H, Dh).transpose(1, 2)   # (B, H, G, Dh)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T, G)

        if vision_mask is not None:
            # vision_mask: True=keep, False=pad
            mask = (~vision_mask).unsqueeze(1).unsqueeze(2)         # (B,1,1,G)
            scores = scores.masked_fill(mask, float('-inf'))

        if self.topk and self.topk < G:
            # sparse softmax over top-k grids per (B,H,T)
            k = self.topk
            topv, topi = scores.topk(k, dim=-1)                     # (B,H,T,k)
            sparse = torch.full_like(scores, float('-inf'))
            sparse.scatter_(-1, topi, topv)                         # keep top-k, -inf elsewhere
            attn = F.softmax(sparse, dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)                        # (B,H,T,G)

        attn = self.drop_attn(attn)

        # ---- 2) Build per-token visual context c_t (average heads) ----
        V = self.v_proj(x_vision)                                   # (B, G, D_ctx) where D_ctx = D_text or d_vision_in
        if self.ctx_to_text is not None:
            V = self.ctx_to_text(V)                                 # (B, G, D_text)

        # contract heads, then grids: (B,H,T,G) × (B,G,D) -> (B,T,D)
        # First average over heads to keep it cheap and stable
        attn_mean = attn.mean(dim=1)                                # (B, T, G)
        c_t = torch.einsum('btg,bgd->btd', attn_mean, V)            # (B, T, D_text)

        # Optional: mask out padded text positions (keep them zero-context)
        if text_mask is not None:
            c_t = c_t * text_mask.unsqueeze(-1)                     # (B,T,1)

        # ---- 3) FiLM around S4 (per-token modulation) ----
        x = self.norm_in(x_text)
        scale_in, shift_in = self.film_in(c_t).chunk(2, dim=-1)     # (B,T,D) each
        x = x * (1 + self.mod_gate * scale_in) + self.mod_gate * shift_in
        # print("X shape before S4:", x.shape)
        # y = self.s4(x)                                              # (B,T,D) temporal mixing by S4
        # print("X shape before S4:", x.shape)
        # y, next_state = self.s4(x)
        if isinstance(self.s4, S4Block) or isinstance(self.s4, S4D):
            y, next_state = self.s4(x.transpose(1, 2))
            y = y.transpose(1, 2)  # (B,T,D)
        else:
            y = self.s4(x)  # (B,T,D)
        # y = y.transpose(1, 2)  # (B,T,D)
        # print("Y shape after S4:", y.shape) 
        scale_out, shift_out = self.film_out(c_t).chunk(2, dim=-1)
        y = y * (1 + self.mod_gate * scale_out) + self.mod_gate * shift_out

        # ---- 4) Residual + FFN ----
        x_text = x_text + y
        x_text = x_text + self.ffn(x_text)
        return self.norm_out(x_text)

def s4_ctor(d_model):
    return S4Block(d_model, l_max=512, bidirectional=False, transposed=False)

def s4d_ctor(d_model):
    return S4D(d_model=d_model, d_state=64, dropout=0.1)

class GridS4Fuser(nn.Module):
    def __init__(
        self,
        d_model=8,
        G=5,
        d_text=1024,
        d_vision_in=768,
        heads=4,
        head_dim=32,
        drop=0.1,
        ffn_mult=4.0,
        proj_vision_to_text=True,
        topk=0,
        ssm_type = "s4",
    ):
        # pass
        super().__init__()
        self.G = G
        # self.s4_ctor = S4Block(d_model, l_max=512, bidirectional=False, transposed=False)
        if ssm_type == "s4":
            self.model = TokenGridCorrFiLMS4(
                d_text=d_text,
                s4_ctor=s4_ctor,
                d_vision_in=d_vision_in,
                heads=heads,
                head_dim=head_dim,
                drop=drop,
                ffn_mult=ffn_mult,
                proj_vision_to_text=proj_vision_to_text,
                topk=topk,
            )
        elif ssm_type == "s4d":
            self.model = TokenGridCorrFiLMS4(
                d_text=d_text,
                s4_ctor=s4d_ctor,
                d_vision_in=d_vision_in,
                heads=heads,
                head_dim=head_dim,
                drop=drop,
                ffn_mult=ffn_mult,
                proj_vision_to_text=proj_vision_to_text,
                topk=topk,
            )

        elif ssm_type == "mamba":
            # from ssm_mamba import MambaBlock
            from mamba_ssm import Mamba
            def mamba_ctor(d_model):
                return Mamba(    
                    d_model=d_model, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=2
                )
            self.model = TokenGridCorrFiLMS4(
                d_text=d_text,
                s4_ctor=mamba_ctor,
                d_vision_in=d_vision_in,
                heads=heads,
                head_dim=head_dim,
                drop=drop,
                ffn_mult=ffn_mult,
                proj_vision_to_text=proj_vision_to_text,
                topk=topk,
            )
        self.importance = nn.Linear(d_text, 1)
        self.softmax = nn.Softmax(dim=0)  # Softmax to normalize weights
    def forward(self, x_text, x_vision):
        # TokenGridCorrFiLMS4 returns (B, T, D_text) - per-token fused embeddings
        # We should return the full sequence, not mean
        fused_tokens = self.model(x_text, x_vision)  # (B, T, D_text)
        return fused_tokens

if __name__ == "__main__":
    # quick test
    # def s4_ctor(d_model):
        # return S4Block(d_model, l_max=512, bidirectional=False, transposed=False)
    model = GridS4Fuser(
        d_text=1024,
        d_vision_in=768,
        heads=4,
        head_dim=32,
        drop=0.1,
        ffn_mult=4.0,
        proj_vision_to_text=True,
        topk=0,
    ).cuda()

    B, T, G = 2, 5, 768
    x_text = torch.randn(B, T, 1024).cuda()
    x_vision = torch.randn(B, G, 768).cuda()
    out = model(x_text, x_vision)
    print(out.shape)  # (B,T,D)
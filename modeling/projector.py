
import torch.nn as nn

from modeling.base import Projector
from modeling.factory import register_projector

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.fc(x)))

@register_projector("residual_ffn", "ResidualFFN")
class ResidualFFNProjector(Projector):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=None,
        dropout=0.1,
        final_activation=False
    ):
        super(ResidualFFNProjector, self).__init__()
        self.projector = MLP(input_dim, output_dim, dropout)
        self.norm1 = nn.LayerNorm(output_dim)
        
        hidden_dim = hidden_dim or output_dim // 2

        self.ffn = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm2 = nn.LayerNorm(output_dim)
        self.final_activation = final_activation
        if final_activation:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.projector(x)
        x = self.norm1(x)

        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)

        if self.final_activation:
            x = self.activation(x)

        return x

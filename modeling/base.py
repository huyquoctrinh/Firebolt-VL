
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class Projector(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Fuser(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

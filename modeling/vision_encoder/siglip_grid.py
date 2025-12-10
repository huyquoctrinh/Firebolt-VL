import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def make_tiles(img: torch.Tensor, tile_size: int = 128, overlap: int = 0):
    """
    Split each image in the batch into grid tiles.
    img: [B,3,H,W]
    return: [N,3,tile_size,tile_size], meta: [(b, y0, x0, h, w, H, W)], counts per image
    """
    B, C, H, W = img.shape
    stride = tile_size - overlap
    ys = list(range(0, max(H - tile_size, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - tile_size, 0) + 1, stride)) or [0]

    tiles, meta, counts = [], [], [0] * B
    for b in range(B):
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + tile_size, H), min(x0 + tile_size, W)
                patch = img[b:b+1, :, y0:y1, x0:x1]
                pad = (0, tile_size - (x1 - x0), 0, tile_size - (y1 - y0))
                patch = F.pad(patch, pad, mode="replicate")
                tiles.append(patch)
                meta.append((b, y0, x0, y1 - y0, x1 - x0, H, W))
                counts[b] += 1
    if tiles:
        tiles = torch.cat(tiles, dim=0)
    else:
        tiles = img.new_zeros((0, C, tile_size, tile_size))
    return tiles, meta, counts


class VisionEncoder(nn.Module):
    """Your SigLIP vision encoder"""
    def __init__(
        self,
        ckpt_path="/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/vision_siglip",
        vision_weight_path=None,
        dtype=torch.float16,
        device=torch.device("cuda"),
    ):
        super().__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(
            ckpt_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if vision_weight_path is not None:
            state_dict = torch.load(vision_weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        # self.model = self.model.to(device)
        # self.model = self.model.to_empty(device=device)

    @torch.no_grad()
    def forward(self, x):
        return self.model.get_image_features(x.to(self.device), interpolate_pos_encoding=True)


class GridVisionEncoder(nn.Module):
    """
    Returns a list of embeddings per image.
      - Each image → list of tile embeddings [num_tiles_i, D]
      - Optional global embedding prepended per image
    """
    def __init__(
        self,
        ckpt_path,
        tile_size: int = 128,
        overlap: int = 2,
        add_global: bool = True,
        global_size: int = 256,
    ):
        super().__init__()
        self.encoder = VisionEncoder(ckpt_path=ckpt_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.add_global = add_global
        self.global_size = global_size

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """
        images: [B,3,H,W]
        return: list of [Ti, D] tensors per image
        """
        device = self.encoder.device
        images = images.to(device)

        # Split into tiles
        tiles, meta, counts = make_tiles(images, self.tile_size, self.overlap)
        tile_feats = self.encoder(tiles) if tiles.numel() > 0 else torch.zeros(0, device=device)

        # Optional global embedding
        global_feats = None
        if self.add_global:
            gimg = F.interpolate(
                images, size=(self.global_size, self.global_size),
                mode="bilinear", align_corners=False
            )
            global_feats = self.encoder(gimg)  # [B, D]

        # Group by batch index
        seqs = []
        idx = 0
        for b in range(images.size(0)):
            n_tiles = counts[b]
            tf = tile_feats[idx:idx + n_tiles]
            idx += n_tiles
            if self.add_global:
                seq = torch.cat([global_feats[b:b+1], tf], dim=0)
            else:
                seq = tf
            seqs.append(seq)
        seqs = torch.stack(seqs, dim=0)
        # print("GridVisionEncoder output seqs shape:", seqs.shape)
        return seqs


if __name__ == "__main__":
    from transformers import AutoProcessor
    from transformers.image_utils import load_image
    model = GridVisionEncoder(ckpt_path="/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/siglip2_base_16_256").to("cuda")
    
    processor = AutoProcessor.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/siglip2_base_16_256")
# load the image
    image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
    inputs = processor(images=[image], return_tensors="pt").to("cuda")
    # print(inputs.keys())
    # run infernece
    print("Input pixel values shape:", inputs["pixel_values"].shape)
    input_pixel_values = inputs["pixel_values"].to("cuda:0")
    with torch.no_grad():
        image_embeddings = model(input_pixel_values)    
    # print(image_embeddings.shape) # 1x1024
    for i, emb in enumerate(image_embeddings):
        print(f"Image {i} has {emb.size(0)} embeddings of dim {emb.size(1)}")
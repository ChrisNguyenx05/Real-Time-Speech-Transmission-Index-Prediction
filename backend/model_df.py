import torch.nn as nn
import torch

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class DirectEncoder(nn.Module):
    def __init__(self, input_dim=5, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, emb_dim),
            nn.ReLU()
        )
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class RayEncoder(nn.Module):
    def __init__(self, input_dim=6, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, emb_dim),
            nn.ReLU()
        )
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class TwoBranchRayNet(nn.Module):
    def __init__(self, direct_dim=5, ray_dim=6, ray_emb_dim=32, output_dim=3):
        super().__init__()

        self.direct_encoder = DirectEncoder(input_dim=direct_dim, emb_dim=32)
        self.ray_encoder = RayEncoder(input_dim=ray_dim, emb_dim=ray_emb_dim)

        self.fusion = nn.Sequential(
            nn.Linear(32 + ray_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)
        )
        self.fusion.apply(init_weights)

    def forward(self, direct, ray):
        B = ray.size(0)

        direct_emb = self.direct_encoder(direct)

        # Trải phẳng ray_encoder xử lý, sau đó cuộn lại thành 14 tia
        ray_flat = ray.view(-1, ray.size(-1))
        ray_emb = self.ray_encoder(ray_flat)
        ray_emb = ray_emb.view(B, 14, -1)

        ray_global = ray_emb.mean(dim=1)

        fused = torch.cat([direct_emb, ray_global], dim=1)
        return self.fusion(fused)
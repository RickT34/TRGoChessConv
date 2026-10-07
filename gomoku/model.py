"""输入 [B,N,N]，输出 [B,N,N] 的落子偏好（logits）。"""

from pathlib import Path
import os

import torch
from torch import nn
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, bias=False),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 5, padding=2, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x):
        return torch.nn.functional.silu(x + self.layers(x))


class ActorCritic(nn.Module):
    """残差策略/价值网络；价值头聚合全盘特征。"""
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        self.architecture, self.channels, self.blocks = "residual", channels, blocks
        self.trunk = nn.Sequential(
            nn.Conv2d(1, channels, 5, padding=2), nn.SiLU(),
            *[ResidualBlock(channels) for _ in range(blocks)],
        )
        self.policy_head = nn.Conv2d(channels, 1, 1)
        self.value_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Linear(channels, channels), nn.SiLU(),
                                        nn.Linear(channels, 1), nn.Tanh())

    def evaluate(self, states):
        features = self.trunk(states.float().unsqueeze(1))
        return self.policy_head(features).squeeze(1), self.value_head(features).squeeze(1)

    def forward(self, states, return_values=False):
        result = self.evaluate(states)
        return result if return_values else result[0]

    def distribution(self, states: torch.Tensor) -> Categorical:
        if (states.flatten(1) != 0).all(1).any():
            raise ValueError("棋盘已满，没有合法动作")
        logits = self(states).masked_fill(states != 0, -torch.inf)
        return Categorical(logits=logits.flatten(1))

    @torch.no_grad()
    def choose(self, state, greedy: bool = False) -> int:
        device = next(self.parameters()).device
        states = torch.as_tensor(state, device=device).unsqueeze(0)
        dist = self.distribution(states)
        return int(dist.probs.argmax(1).item() if greedy else dist.sample().item())


def save_policy(path, model: ActorCritic, size: int, config: dict, **training):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "channels": model.channels,
               "architecture": model.architecture, "blocks": getattr(model, "blocks", 0),
               "size": size, "config": config, "format_version": 2, **training}
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)  # Web 端不会读到正在写入的半个 checkpoint。


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "residual":
        raise ValueError("仅支持带策略头和价值头的残差模型")
    model = ActorCritic(checkpoint["channels"], checkpoint["blocks"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint

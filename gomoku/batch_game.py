"""多盘棋在同一设备上更新。训练热路径不把每步棋盘搬回 CPU。"""

import torch


class BatchGomoku:
    def __init__(self, batch=128, size=19, device="cpu"):
        if batch < 1 or size < 5:
            raise ValueError("batch >= 1，size >= 5")
        self.size, self.batch = size, batch
        self.board = torch.zeros((batch, size, size), dtype=torch.int8, device=device)
        self.players = torch.ones(batch, dtype=torch.int8, device=device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.winners = torch.zeros_like(self.players)
        self.lengths = torch.zeros(batch, dtype=torch.long, device=device)
        self.rows = torch.arange(batch, device=device)
        # 只检查最后落子附近的四条九格线，比反复扫描全盘更省计算。
        offsets = torch.arange(-4, 5, device=device)
        directions = torch.tensor([[0, 1], [1, 0], [1, 1], [1, -1]], device=device)
        self.dr = directions[:, :1] * offsets
        self.dc = directions[:, 1:] * offsets

    def observation(self):
        return self.board * self.players[:, None, None]

    def step(self, actions, validate=False):
        """已结束的棋局保持不变；训练采样器保证合法性，可跳过同步校验。"""
        active = ~self.done
        flat = self.board.flatten(1)
        if validate:
            if actions.shape != (self.batch,) or actions.dtype != torch.long:
                raise ValueError("动作须为 [batch] 的 int64 张量")
            if ((actions < 0) | (actions >= self.size**2)).any():
                raise ValueError("落子越界")
            if (active & (flat[self.rows, actions] != 0)).any():
                raise ValueError("落在已有棋子的位置")
        previous = flat[self.rows, actions]
        flat[self.rows, actions] = torch.where(active, self.players, previous)
        self.lengths += active.long()
        r = actions[:, None, None] // self.size + self.dr
        c = actions[:, None, None] % self.size + self.dc
        valid = (r >= 0) & (r < self.size) & (c >= 0) & (c < self.size)
        indices = (r.clamp(0, self.size-1) * self.size + c.clamp(0, self.size-1)).flatten(1)
        lines = flat.gather(1, indices).view(self.batch, 4, 9)
        matches = (lines == self.players[:, None, None]) & valid
        won = matches.unfold(-1, 5, 1).all(-1).any(-1).any(-1) & active
        self.winners = torch.where(won, self.players, self.winners)
        self.done |= won | (self.lengths == self.size**2)
        self.players = torch.where(active, -self.players, self.players)
        return self.done & active

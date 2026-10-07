"""棋盘规则：1 先手，-1 后手，0 空位；无禁手，五连及以上获胜。"""

import numpy as np
import torch
from torch.nn import functional as F


# 水平、垂直、两条对角线。分别使用不同形状，避免漏掉棋盘边缘。
KERNELS = (
    torch.ones(1, 1, 1, 5),
    torch.ones(1, 1, 5, 1),
    torch.stack((torch.eye(5), torch.eye(5).flip(1))).unsqueeze(1),
)


def winner(board: np.ndarray) -> int:
    """卷积和为 +5 / -5，等价于该方向连续五个 +1 / -1。"""
    x = torch.as_tensor(board, dtype=torch.float32)[None, None]
    for kernel in KERNELS:
        sums = F.conv2d(x, kernel)
        if (sums == 5).any():
            return 1
        if (sums == -5).any():
            return -1
    return 0


class Gomoku:
    def __init__(self, size: int = 19):
        if size < 5:
            raise ValueError("棋盘边长至少为 5")
        self.size = size
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.to_play = 1
        self.winner = 0
        self.done = False
        return self.observation()

    def observation(self) -> np.ndarray:
        """当前行棋方视角：己方始终是 +1，对方始终是 -1。"""
        return self.board.copy() * self.to_play

    def legal_actions(self) -> np.ndarray:
        if self.done:
            return np.array([], dtype=np.int64)
        return np.flatnonzero(self.board.ravel() == 0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """action = row * size + col；reward 属于刚刚落子的一方。

        非终局返回的是对手的 observation；终局之后不可继续落子。
        这里只给最后一手 +1，自我对弈训练再给双方整局动作分配胜负回报。
        """
        if self.done:
            raise ValueError("对局已结束，请 reset")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.size**2:
            raise ValueError("落子位置超出棋盘")
        row, col = divmod(int(action), self.size)
        if self.board[row, col] != 0:
            raise ValueError("该位置已有棋子")
        self.board[row, col] = self.to_play
        self.winner = winner(self.board)
        self.done = bool(self.winner or not (self.board == 0).any())
        reward = float(self.winner == self.to_play)
        self.to_play *= -1
        return self.observation(), reward, self.done

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        header = "   " + " ".join(f"{i:2}" for i in range(1, self.size + 1))
        rows = [f"{r + 1:2} " + " ".join(f" {symbols[int(x)]}" for x in row)
                for r, row in enumerate(self.board)]
        return "\n".join([header, *rows])

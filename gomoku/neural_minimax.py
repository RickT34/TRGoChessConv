"""独立的神经网络 Minimax：Negamax + Alpha-Beta，叶节点使用价值头。

top_p=1 搜索所有合法动作；top_p<1 按策略概率累计选取候选，额外保留
所有一步成五和防五落点。局面分数只来自规则或价值头。
时间限制为软限制：至少完成一层，之后只采用完整完成的迭代。
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import time

import numpy as np
import torch

from .game import winner


WIN = 2.0  # 严格高于 Tanh 价值头的 [-1, 1]，终局不能被网络推翻。


@dataclass(frozen=True)
class SearchConfig:
    depth: int = 2
    top_p: float = 0.9
    time_limit: float = 3.0

    def __post_init__(self):
        if type(self.depth) is not int or not 1 <= self.depth <= 6:
            raise ValueError("搜索深度须为 1 到 6 的整数（每层为一手棋）")
        if (type(self.top_p) not in (int, float) or not math.isfinite(self.top_p)
                or not 0 < self.top_p <= 1):
            raise ValueError("Top-p 须在 (0, 1] 内；1 表示全部合法落点")
        if (type(self.time_limit) not in (int, float) or not math.isfinite(self.time_limit)
                or not 0 <= self.time_limit <= 30):
            raise ValueError("搜索时限须在 0 到 30 秒内；0 表示不限时")

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return cls()
        if not isinstance(data, dict) or set(data) - {"depth", "top_p", "time_limit"}:
            raise ValueError("搜索设置仅支持 depth、top_p、time_limit")
        return cls(**data)


@lru_cache(maxsize=21)
def _lines(size):
    return np.asarray([
        [(r + i * dr) * size + c + i * dc for i in range(5)]
        for r in range(size) for c in range(size)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
        if 0 <= r + 4 * dr < size and 0 <= c + 4 * dc < size
    ])


def _won(board, action, player):
    """只检查刚落下的一子，不调用网络或全盘卷积。"""
    size = len(board)
    row, col = divmod(action, size)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                count += 1
                r, c = r + sign * dr, c + sign * dc
        if count >= 5:
            return True
    return False


class _Timeout(Exception):
    pass


class NeuralMiniMax:
    """传入冻结的 ActorCritic；search 不修改棋盘或模型权重。

    输入棋盘使用绝对颜色（黑=1，白=-1），player 是当前行棋方。
    返回的 score 始终属于根节点行棋方，不能解释为胜率。
    同一实例的 search 应串行调用。
    """

    def __init__(self, model, config=None):
        if not callable(getattr(model, "evaluate", None)):
            raise ValueError("神经网络搜索需要带价值头的残差 ActorCritic 模型；该模型只有策略头")
        self.model = model
        self.config = config or SearchConfig()

    def _network(self, board, player):
        key = (board.tobytes(), player)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        device = next(self.model.parameters()).device
        state = torch.as_tensor(board * player, device=device)[None]
        logits, values = self.model.evaluate(state)
        logits = logits[0].flatten().float().cpu().numpy()
        value = float(values[0].item())
        if not np.isfinite(logits).all() or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("模型输出无效：需要有限策略 logits 和 [-1, 1] 内的价值评分")
        self.stats["evaluations"] += 1
        self.cache[key] = (logits, value)
        return logits, value

    def _actions(self, board, player):
        legal = np.flatnonzero(board.ravel() == 0)
        logits, _ = self._network(board, player)
        # 先按纯策略概率确定最短前缀，再进行战术排序；占用位置不参与归一化。
        ordered = legal[np.argsort(-logits[legal], kind="stable")]
        if self.config.top_p < 1:
            scores = logits[ordered].astype(np.float64)
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            count = int(np.searchsorted(probabilities.cumsum(), self.config.top_p, side="left")) + 1
            ordered = ordered[:count]
        lines = _lines(len(board))
        stones = board.ravel()[lines]
        empty = (stones == 0).sum(1) == 1
        wins, blocks = [], []
        for side, target in ((player, wins), (-player, blocks)):
            selected = lines[empty & ((stones == side).sum(1) == 4)]
            target.extend(np.unique(selected[board.ravel()[selected] == 0]).tolist())
        mandatory = set(wins) | set(blocks)
        actions = sorted(set(ordered.tolist()) | mandatory, key=lambda a: (
            0 if a in wins else 1 if a in blocks else 2, -float(logits[a]), a))
        return actions

    def _search(self, board, player, depth, alpha, beta, last_action):
        if self.enforce_deadline and time.perf_counter() >= self.deadline:
            raise _Timeout
        self.stats["nodes"] += 1
        if _won(board, last_action, -player):
            return -WIN
        if not (board == 0).any():
            return 0.0
        if depth == 0:
            return self._network(board, player)[1]
        best = -math.inf
        for action in self._actions(board, player):
            board.flat[action] = player
            try:
                score = -self._search(board, -player, depth - 1, -beta, -alpha, action)
            finally:
                board.flat[action] = 0
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                self.stats["cutoffs"] += 1
                break
        return best

    @torch.inference_mode()
    def search(self, board, player):
        source = np.asarray(board)
        if (source.ndim != 2 or source.shape[0] != source.shape[1]
                or not 5 <= len(source) <= 25 or not np.isin(source, [-1, 0, 1]).all()):
            raise ValueError("棋盘须为 5 到 25 阶方阵，棋子取值只能是 -1、0、1")
        if isinstance(player, (bool, np.bool_)) or player not in (-1, 1):
            raise ValueError("当前行棋方须为 1 或 -1")
        board = source.astype(np.int8, copy=True)
        if winner(board) or not (board == 0).any():
            raise ValueError("不能在终局上搜索")
        start = time.perf_counter()
        self.deadline = start + self.config.time_limit if self.config.time_limit else math.inf
        self.stats = dict(nodes=0, cutoffs=0, evaluations=0, cache_hits=0)
        self.cache = {}
        was_training = self.model.training
        self.model.eval()
        try:
            actions = self._actions(board, player)
            completed_depth, timed_out = 0, False
            best_action, best_score = actions[0], -math.inf
            for depth in range(1, self.config.depth + 1):
                # 一层必定完整完成；不使用策略 argmax 冒充搜索结果。
                self.enforce_deadline = depth > 1
                alpha, iteration_action = -math.inf, actions[0]
                try:
                    for action in actions:
                        board.flat[action] = player
                        try:
                            score = -self._search(board, -player, depth - 1, -math.inf, -alpha, action)
                        finally:
                            board.flat[action] = 0
                        if score > alpha:
                            alpha, iteration_action = score, action
                except _Timeout:
                    timed_out = True
                    break
                best_action, best_score, completed_depth = iteration_action, alpha, depth
                actions.remove(best_action)
                actions.insert(0, best_action)
            return dict(action=best_action, score=best_score, depth=completed_depth,
                        requested_depth=self.config.depth, timed_out=timed_out,
                        milliseconds=round((time.perf_counter() - start) * 1000, 2),
                        config=asdict(self.config), **self.stats)
        finally:
            self.model.train(was_training)
            self.cache.clear()

    def choose(self, board, player):
        return self.search(board, player)["action"]

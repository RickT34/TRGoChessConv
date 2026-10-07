"""Web Server 的模型推理和棋局状态。"""

import hashlib
from dataclasses import asdict
import io
import time
from pathlib import Path

import torch

from .game import Gomoku
from .model import load_policy
from .neural_minimax import NeuralMiniMax, SearchConfig


class GameSession:
    """每个浏览器一盘棋；模型在本局中保持冻结，悔棋重放已有棋谱。"""

    def __init__(self, model_path, model_name, human=1, mode="greedy", seed=0, device="cpu", auto_reply=True,
                 search=None):
        if type(human) is not int or human not in (-1, 1):
            raise ValueError("请选择执黑或执白")
        if mode not in ("greedy", "sample", "minimax"):
            raise ValueError("未知的落子模式")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("随机种子须为 0 到 4294967295 的整数")
        model_bytes = Path(model_path).read_bytes()
        self.model, checkpoint = load_policy(io.BytesIO(model_bytes), device)
        self.model.eval()
        self.searcher = NeuralMiniMax(self.model, SearchConfig.from_dict(search)) if mode == "minimax" else None
        self.env = Gomoku(checkpoint["size"])
        self.human, self.mode, self.seed = human, mode, seed
        self.model_name = model_name
        self.metadata = {"model": model_name, "sha256": hashlib.sha256(model_bytes).hexdigest(),
                         "training_config": checkpoint.get("config", {}), "channels": checkpoint["channels"]}
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.history = []
        self.version = 0
        if auto_reply:
            self._ai_turn()

    @torch.inference_mode()
    def probabilities(self):
        state = torch.as_tensor(self.env.observation(), device=next(self.model.parameters()).device)[None]
        return self.model.distribution(state).probs[0]

    def _place(self, action, decision=None):
        player = self.env.to_play
        self.env.step(action)
        row, col = divmod(action, self.env.size)
        self.history.append({"action": action, "row": row, "col": col,
                             "player": player, "decision": decision})

    def _ai_turn(self):
        if self.env.done or self.env.to_play == self.human:
            return
        start = time.perf_counter()
        probs = self.probabilities()
        search_result = None
        if self.searcher:
            search_result = self.searcher.search(self.env.board, self.env.to_play)
            action = search_result["action"]
        elif self.mode == "greedy":
            action = int(probs.argmax().item())
        else:
            action = int(torch.multinomial(probs, 1, generator=self.generator).item())
        values, indices = probs.topk(min(5, len(self.env.legal_actions())))
        decision = {"probability": float(probs[action].item()),
                    "milliseconds": round((time.perf_counter() - start) * 1000, 2),
                    "top": [{"action": int(a), "probability": float(p)}
                            for a, p in zip(indices.tolist(), values.tolist())]}
        if search_result is not None:
            decision["search"] = search_result
        self._place(action, decision)

    def move(self, action, version):
        self._check_version(version)
        if self.env.done:
            raise ValueError("本局已结束，请重新开始或悔棋")
        if self.env.to_play != self.human:
            raise ValueError("还未轮到你落子")
        if type(action) is not int:
            raise ValueError("落子位置必须是整数")
        previous_moves = len(self.history)
        self._place(action)
        try:
            self._ai_turn()
        except Exception:
            # 推理失败时回滚本轮，避免请求报错后棋盘卡在 AI 回合。
            self.history = self.history[:previous_moves]
            self._replay()
            raise
        self.version += 1

    def _check_version(self, version):
        if type(version) is not int or version != self.version:
            raise ValueError("棋盘已更新，请刷新后再操作")

    def undo(self, version):
        self._check_version(version)
        human_moves = [i for i, move in enumerate(self.history) if move["player"] == self.human]
        if not human_moves:
            raise ValueError("还没有可以撤回的落子")
        self.history = self.history[:human_moves[-1]]
        self._replay()
        # 保留采样器当前位置：悔棋后同一局面可以探索不同的随机应手。
        self.version += 1

    def _replay(self):
        self.env.reset()
        for move in self.history:
            self.env.step(move["action"])

    def snapshot(self):
        probabilities = None if self.env.done else self.probabilities().cpu().tolist()
        return {"board": self.env.board.tolist(), "size": self.env.size, "human": self.human,
                "to_play": self.env.to_play, "done": self.env.done, "winner": self.env.winner,
                "version": self.version, "history": self.history, "probabilities": probabilities,
                "model": self.model_name, "mode": self.mode, "seed": self.seed,
                "search": asdict(self.searcher.config) if self.searcher else None,
                "can_undo": any(move["player"] == self.human for move in self.history)}

    def export(self):
        return {"format_version": 1, "size": self.env.size, "human": self.human,
                "mode": self.mode, "seed": self.seed, "done": self.env.done, "winner": self.env.winner,
                "search": asdict(self.searcher.config) if self.searcher else None,
                "rules": "freestyle: five or more; no forbidden moves",
                "coordinates": "zero-based row and col; action = row * size + col",
                "model": self.metadata, "moves": self.history}

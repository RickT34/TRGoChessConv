"""TRGoChessC 棋型教师的 ctypes 接口；OpenMP 并行搜索独立局面。"""

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np

SOURCE = Path(__file__).with_name("native") / "teacher.c"
PATTERNS = ("210", "010", "2110", "0110", "21110", "01110", "21010", "01010",
            "211010", "210110", "010110", "211110", "011110", "10111", "11011", "11111")
WEIGHTS = (1.147176, 1.017057, 2.872098, 3.774109, 14.642408, 39.582565, .1, .1,
           17.334492, 16.255480, 24.932312, 31.667524, 11747.007813, 18.771877,
           26.783592, 1e10)


def build_library():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    directory = SOURCE.parents[2] / "build"
    directory.mkdir(exist_ok=True)
    target = directory / f"teacher-{digest}.so"
    if not target.exists():
        temporary = target.with_suffix(f".{os.getpid()}.tmp.so")
        subprocess.run(["gcc", "-O3", "-std=c11", "-Wall", "-Wextra", "-shared", "-fPIC",
                        "-fopenmp", str(SOURCE), "-o", str(temporary)], check=True)
        temporary.replace(target)
    return target


class CTeacher:
    def __init__(self, depth=4, width=16, nodes=20000, workers=8):
        if not 1 <= depth <= 12 or width < 0 or nodes < 0 or workers < 1:
            raise ValueError("depth 须为 1..12，width/nodes 非负，workers 为正数")
        self.depth, self.width, self.nodes, self.workers = depth, width, nodes, workers
        self.library = ctypes.CDLL(str(build_library()))
        array = np.ctypeslib.ndpointer
        self.library.teacher_batch.argtypes = [array(np.int8, flags="C_CONTIGUOUS"),
            array(np.int32, flags="C_CONTIGUOUS"), *([ctypes.c_int] * 6),
            array(np.int32), array(np.float64), array(np.int32), array(np.int32)]
        self.library.teacher_batch.restype = None
        self.library.teacher_policy_batch.argtypes = [array(np.int8, flags="C_CONTIGUOUS"),
            array(np.int32, flags="C_CONTIGUOUS"), *([ctypes.c_int] * 6),
            array(np.float64, flags="C_CONTIGUOUS"), array(np.int32), array(np.int32)]
        self.library.teacher_policy_batch.restype = None
        self.library.teacher_score.argtypes = [array(np.int8, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]
        self.library.teacher_score.restype = ctypes.c_double

    @staticmethod
    def validate(boards, players):
        boards, players = np.asarray(boards), np.asarray(players)
        if boards.ndim != 3 or boards.shape[1] != boards.shape[2] or not 5 <= boards.shape[1] <= 25:
            raise ValueError("教师接收 [batch, N, N] 棋盘，N 为 5..25")
        if players.shape != (len(boards),) or not np.isin(players, [-1, 1]).all():
            raise ValueError("每盘棋必须指定 +1 或 -1 行棋方")
        if not np.isin(boards, [-1, 0, 1]).all():
            raise ValueError("棋盘仅可包含 -1、0、1")
        return np.ascontiguousarray(boards, dtype=np.int8), np.ascontiguousarray(players, dtype=np.int32)

    def choose_batch(self, boards, players):
        boards, players = self.validate(boards, players)
        count, size, _ = boards.shape
        actions, nodes, depths = (np.empty(count, dtype=np.int32) for _ in range(3))
        scores = np.empty(count, dtype=np.float64)
        self.library.teacher_batch(boards, players, count, size, self.depth, self.width,
                                   self.nodes, self.workers, actions, scores, nodes, depths)
        if (actions < 0).any():
            raise ValueError("教师收到终局棋盘，或搜索内存不足")
        return actions, {"nodes": nodes, "depths": depths, "scores": scores}

    def choose(self, board, player):
        return int(self.choose_batch(np.asarray(board)[None], [player])[0][0])

    def policy_batch(self, boards, players, temperature=100.):
        """返回 [B,N*N] 软标签及同深度根评分；温度采用 G11 分数单位。

        只覆盖教师候选点，其他点概率为零；不是全盘完备搜索或获胜概率。
        stats.depths == 0 表示节点预算不足，使用完整静态评分回退。
        """
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('distillation_temperature 必须是有限正数')
        boards, players = self.validate(boards, players)
        count, size, _ = boards.shape
        scores = np.empty((count, size*size), dtype=np.float64)
        nodes, depths = (np.empty(count, dtype=np.int32) for _ in range(2))
        self.library.teacher_policy_batch(boards, players, count, size, self.depth, self.width,
                                          self.nodes, self.workers, scores, nodes, depths)
        if not np.isfinite(scores).any(axis=1).all():
            raise ValueError('教师收到终局棋盘，或搜索内存不足')
        with np.errstate(over='ignore', under='ignore'):
            weights = np.exp((scores - scores.max(axis=1, keepdims=True)) / temperature)
        policies = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)
        return policies, {'nodes': nodes, 'depths': depths, 'scores': scores}

    def score(self, board, player):
        boards, players = self.validate(np.asarray(board)[None], [player])
        return self.library.teacher_score(boards[0], boards.shape[1], int(players[0]))

    def config(self):
        return {"teacher": "trgoc-pattern-g11", "depth": self.depth, "width": self.width,
                "node_budget": self.nodes, "workers": self.workers,
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}

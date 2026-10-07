import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gomoku.game import Gomoku, winner


torch.set_num_threads(1)


class RulesTests(unittest.TestCase):
    def test_all_directions_both_colors_and_edges(self):
        for player in (1, -1):
            for row, col, dr, dc in ((18, 14, 0, 1), (14, 18, 1, 0),
                                     (14, 14, 1, 1), (14, 4, 1, -1),
                                     (0, 0, 0, 1), (0, 0, 1, 0)):
                board = np.zeros((19, 19), dtype=np.int8)
                for i in range(5):
                    board[row + i * dr, col + i * dc] = player
                with self.subTest(player=player, direction=(dr, dc), start=(row, col)):
                    self.assertEqual(winner(board), player)

    def test_gaps_mixed_colors_overlines_and_diagonal_boundary(self):
        board = np.zeros((19, 19), dtype=np.int8)
        board[18, 10:16] = 1
        self.assertEqual(winner(board), 1)
        board[18, 12] = 0
        self.assertEqual(winner(board), 0)
        board[18, 12] = -1
        self.assertEqual(winner(board), 0)
        board[:] = 0
        board[0, 15:19] = 1
        board[1, 0] = 1  # 不可将展平矩阵的跨行位置当作连续五子。
        self.assertEqual(winner(board), 0)

    def test_turns_illegal_moves_and_terminal_reward(self):
        env = Gomoku()
        obs, reward, done = env.step(0)
        self.assertEqual(obs[0, 0], -1)
        self.assertEqual((reward, done, env.to_play), (0, False, -1))
        obs[:] = 0
        self.assertEqual(env.board[0, 0], 1)
        for illegal in (0, -1, 361, 1.5):
            before = env.board.copy()
            with self.assertRaises(ValueError):
                env.step(illegal)
            np.testing.assert_array_equal(env.board, before)
            self.assertEqual(env.to_play, -1)
        for action in (19, 1, 20, 2, 21, 3, 22, 4):
            _, reward, done = env.step(action)
        self.assertEqual((env.winner, reward, done), (1, 1, True))
        self.assertEqual(len(env.legal_actions()), 0)
        with self.assertRaises(ValueError):
            env.step(30)
        self.assertFalse(env.reset().any())

    def test_full_board_draw(self):
        # 13 枚黑棋、12 枚白棋，所有行、列、对角线均无五连。
        final = np.array([[1, 1, -1, -1, 1], [-1, -1, 1, 1, -1],
                          [1, 1, -1, -1, 1], [-1, -1, 1, 1, -1],
                          [1, -1, 1, -1, 1]], dtype=np.int8)
        self.assertEqual(winner(final), 0)
        env = Gomoku(5)
        black = np.flatnonzero(final.ravel() == 1)
        white = np.flatnonzero(final.ravel() == -1)
        for i in range(12):
            env.step(int(black[i]))
            env.step(int(white[i]))
        _, reward, done = env.step(int(black[-1]))
        self.assertEqual((reward, done, env.winner), (0, True, 0))

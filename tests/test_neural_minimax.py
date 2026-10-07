"""与无剪枝穷举对照，并检查视角、终局、超时和 Web 棋谱。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from gomoku.game import winner
from gomoku.model import ActorCritic, save_policy
from gomoku.neural_minimax import NeuralMiniMax, SearchConfig, WIN, _won
from gomoku.web import App


torch.set_num_threads(1)


class ToyValue(torch.nn.Module):
    """策略刻意偏好左上角，价值则偏好中心及右下，测试两头没有混淆。"""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def evaluate(self, states):
        weights = torch.arange(1, states.shape[-1] ** 2 + 1, device=states.device).reshape(states.shape[-2:])
        logits = -weights.float().expand_as(states)
        value = torch.tanh((states * weights).flatten(1).sum(1) / 100)
        return logits, value


def exhaustive(board, player, depth, model):
    win = winner(board)
    if win:
        return WIN * win * player
    if not (board == 0).any():
        return 0.0
    if depth == 0:
        return model.evaluate(torch.as_tensor(board * player)[None])[1][0].item()
    scores = []
    for action in np.flatnonzero(board.ravel() == 0):
        board.flat[action] = player
        scores.append(-exhaustive(board, -player, depth - 1, model))
        board.flat[action] = 0
    return max(scores)


class NeuralMiniMaxTests(unittest.TestCase):
    def test_top_p_selects_shortest_prefix_and_masks_occupied_cells(self):
        board = np.zeros((5, 5), dtype=np.int8)
        board.flat[0] = 1
        logits = np.full(25, -1000.0)
        logits[0] = 1000  # 已落子位置不能占据概率质量。
        logits[1:5] = np.log([0.55, 0.25, 0.15, 0.05])
        for top_p, expected in ((0.5, [1]), (0.8, [1, 2]), (0.9, [1, 2, 3]),
                                (1, list(range(1, 25)))):
            engine = NeuralMiniMax(ToyValue(), SearchConfig(top_p=top_p))
            with patch.object(engine, '_network', return_value=(logits, 0)):
                self.assertEqual(engine._actions(board, -1), expected)

    def test_top_p_boundary_ties_and_single_legal_move(self):
        board = np.zeros((5, 5), dtype=np.int8)
        logits = np.full(25, -1000.0)
        logits[:2] = 0  # 两个落点概率各 50%，恰好达到阈值时选第一个。
        engine = NeuralMiniMax(ToyValue(), SearchConfig(top_p=0.5))
        with patch.object(engine, '_network', return_value=(logits, 0)):
            self.assertEqual(engine._actions(board, 1), [0])
        board[:] = 1
        board.flat[12] = 0
        with patch.object(engine, '_network', return_value=(np.zeros(25), 0)):
            self.assertEqual(engine._actions(board, -1), [12])

    def test_top_p_is_selected_before_adding_tactical_moves(self):
        board = np.zeros((9, 9), dtype=np.int8)
        board[8, :4] = -1
        logits = np.full(81, -1000.0)
        logits[:2] = 0
        engine = NeuralMiniMax(ToyValue(), SearchConfig(top_p=0.9))
        with patch.object(engine, '_network', return_value=(logits, 0)):
            self.assertEqual(engine._actions(board, 1), [76, 0, 1])

    def test_alpha_beta_matches_exhaustive_both_colors_and_depths(self):
        model = ToyValue()
        rng = np.random.default_rng(17)
        for _ in range(3):
            while True:
                board = rng.choice([-1, 1], (5, 5)).astype(np.int8)
                board.flat[rng.choice(25, 6, replace=False)] = 0
                if not winner(board):
                    break
            original = board.copy()
            for player in (-1, 1):
                for depth in (1, 2, 3):
                    search = NeuralMiniMax(model, SearchConfig(depth, 1, 0))
                    result = search.search(board, player)
                    expected = exhaustive(board.copy(), player, depth, model)
                    self.assertAlmostEqual(result['score'], expected)
                    child = board.copy()
                    child.flat[result['action']] = player
                    self.assertAlmostEqual(-exhaustive(child, -player, depth - 1, model), expected)
                    self.assertEqual(result['depth'], depth)
                    np.testing.assert_array_equal(board, original)
            self.assertTrue(model.training)

    def test_uses_value_head_and_alternates_perspective(self):
        board = np.zeros((5, 5), dtype=np.int8)
        for side in (-1, 1):
            result = NeuralMiniMax(ToyValue(), SearchConfig(1, 1, 0)).search(board, side)
            self.assertEqual(result['action'], 24)  # 策略 argmax 却是 0。
            self.assertAlmostEqual(result['score'], np.tanh(.25), places=6)

    def test_win_and_forced_block_survive_low_top_p(self):
        for side in (-1, 1):
            for attack in (True, False):
                board = np.zeros((9, 9), dtype=np.int8)
                board[8, :4] = side if attack else -side
                result = NeuralMiniMax(ToyValue(), SearchConfig(2, 0.01, 0)).search(board, side)
                self.assertEqual(result['action'], 76)
                if attack:
                    self.assertEqual(result['score'], WIN)

    def test_pruning_and_cache_are_used(self):
        result = NeuralMiniMax(ToyValue(), SearchConfig(3, 0.999, 0)).search(np.zeros((5, 5)), 1)
        self.assertGreater(result['cutoffs'], 0)
        self.assertGreater(result['cache_hits'], 0)

    def test_timeout_keeps_complete_first_iteration(self):
        board = np.zeros((5, 5), dtype=np.int8)
        with patch('gomoku.neural_minimax.time.perf_counter', side_effect=[0, 10, 10]):
            result = NeuralMiniMax(ToyValue(), SearchConfig(4, 1, 1)).search(board, 1)
        self.assertEqual(result['depth'], 1)
        self.assertEqual(result['action'], 24)
        self.assertTrue(result['timed_out'])
        self.assertFalse(board.any())

    def test_draw_and_terminal_rejection(self):
        rng = np.random.default_rng(20)
        while True:
            board = rng.choice([-1, 1], (5, 5)).astype(np.int8)
            if not winner(board):
                break
        search = NeuralMiniMax(ToyValue(), SearchConfig(2, 1, 0))
        with self.assertRaises(ValueError):
            search.search(board, 1)
        side = int(board[0, 0])
        board[0, 0] = 0
        result = search.search(board, side)
        self.assertEqual(result['action'], 0)
        self.assertEqual(result['score'], 0)
        board[0, :] = 1
        with self.assertRaises(ValueError):
            search.search(board, -1)

    def test_last_move_detection_matches_rules_edges_diagonals_overline(self):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for player in (-1, 1):
                board = np.zeros((9, 9), dtype=np.int8)
                r, c = 0, 8 if dc < 0 else 0
                for i in range(6):
                    action = (r + i * dr) * 9 + c + i * dc
                    board.flat[action] = player
                    self.assertEqual(_won(board, action, player), winner(board) == player)

    def test_invalid_options_and_policy_only_checkpoint(self):
        for options in ({'depth': True}, {'depth': 0}, {'top_p': 0}, {'top_p': -1}, {'top_p': 1.01}, {'top_p': True}, {'top_p': float('nan')}, {'top_p': float('inf')}, {'width': 4}, {'time_limit': float('nan')},
                        {'time_limit': True}, {'unexpected': 1}, []):
            with self.assertRaises(ValueError):
                SearchConfig.from_dict(options)
        with self.assertRaisesRegex(ValueError, '价值头'):
            NeuralMiniMax(torch.nn.Identity())

    def test_web_move_undo_export_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            model = ActorCritic(8, 1).eval()
            path = Path(directory) / 'value.pt'
            save_policy(path, model, 5, {'test': True})
            original = path.read_bytes()
            for app in (App(directory),):
                for human in (-1, 1):
                    config = {'depth': 2, 'top_p': 0.8, 'time_limit': 0}
                    state = app.request('/api/new', {'model': 'value.pt', 'human': human,
                                                   'mode': 'minimax', 'search': config})
                    args = {'session': state['session']}
                    opening = len(state['history'])
                    action = int(np.flatnonzero(np.asarray(state['board']).ravel() == 0)[0])
                    state = app.request('/api/move', {**args, 'version': 0, 'action': action})
                    self.assertEqual(len(state['history']), opening + 2)
                    search = state['history'][-1]['decision']['search']
                    self.assertEqual(search['depth'], 2)
                    self.assertEqual(search['action'], state['history'][-1]['action'])
                    self.assertEqual(state['search'], config)
                    record = app.request('/api/export', args)
                    json.dumps(record, allow_nan=False)
                    self.assertEqual(record['search'], config)
                    self.assertEqual(len(record['model']['sha256']), 64)
                    state = app.request('/api/undo', {**args, 'version': 1})
                    self.assertEqual(len(state['history']), opening)
                    self.assertEqual(app.request('/api/state', args)['search'], config)
                    # 值头／搜索推理报错时，不留下用户已落子但 AI 未应手的半轮。
                    game = app.sessions[state['session']]
                    previous = game.env.board.copy()
                    previous_version = game.version
                    action = int(game.env.legal_actions()[0])
                    with patch.object(game.searcher, 'search', side_effect=ValueError('invalid value')):
                        with self.assertRaisesRegex(ValueError, 'invalid value'):
                            game.move(action, previous_version)
                    np.testing.assert_array_equal(game.env.board, previous)
                    self.assertEqual(len(game.history), opening)
                    self.assertEqual(game.version, previous_version)
                    self.assertEqual(game.env.to_play, human)
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()

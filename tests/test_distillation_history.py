from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from gomoku.config import read_yaml, write_yaml
from gomoku.data import ExpertDataset
from gomoku.game import winner
from gomoku.history import HistoryPool, collect_history_games
from gomoku.model import ActorCritic
from gomoku.rollout import policy_statistics
from gomoku.teacher import CTeacher
from gomoku.train import augment


torch.set_num_threads(1)


class DistributionTests(unittest.TestCase):
    def test_all_immediate_wins_and_forced_block_for_both_colors(self):
        teacher = CTeacher(depth=3, width=1, nodes=20, workers=2)
        for player in (-1, 1):
            board = np.zeros((9, 9), np.int8)
            board[4, 2:6] = player
            policy, stats = teacher.policy_batch(board[None], [player])
            expected = np.zeros(81)
            expected[4*9+1] = expected[4*9+6] = .5
            np.testing.assert_allclose(policy[0], expected)
            self.assertEqual(stats['depths'][0], 1)
            board[4, 1] = -player
            policy, _ = teacher.policy_batch(board[None], [-player])
            self.assertEqual(policy[0, 4*9+6], 1.)
            self.assertEqual(np.count_nonzero(policy), 1)

    def test_root_scores_match_exhaustive_two_ply_candidate_values(self):
        teacher = CTeacher(depth=2, width=0, nodes=0, workers=1)
        rng = np.random.default_rng(17)
        checked = 0
        while checked < 10:
            board = rng.choice([-1, 1], (5, 5)).astype(np.int8)
            board.flat[rng.choice(25, 6, replace=False)] = 0
            if winner(board):
                continue
            player = 1 if checked % 2 else -1
            policy, stats = teacher.policy_batch(board[None], [player])
            for action in np.flatnonzero(np.isfinite(stats['scores'][0])):
                board.flat[action] = player
                if winner(board) == player:
                    expected = 1e12-1
                else:
                    replies = []
                    empty = np.flatnonzero(board.ravel() == 0)
                    immediate, blocks = [], []
                    for reply in empty:
                        board.flat[reply] = -player
                        if winner(board) == -player:
                            immediate.append(reply)
                        board.flat[reply] = player
                        if winner(board) == player:
                            blocks.append(reply)
                        board.flat[reply] = 0
                    # The teacher's selective tree forces wins/blocks, even at
                    # its last expanded ply. Enumerate that same legal set.
                    for reply in (immediate or blocks or empty.tolist()):
                        board.flat[reply] = -player
                        replies.append(-1e12+2 if winner(board) == -player else teacher.score(board, player))
                        board.flat[reply] = 0
                    expected = min(replies)
                board.flat[action] = 0
                self.assertAlmostEqual(stats['scores'][0, action], expected, delta=1e-3)
            self.assertAlmostEqual(float(policy.sum()), 1., places=6)
            checked += 1

    def test_budget_discards_partial_pass_and_threads_agree(self):
        board = np.zeros((9, 9), np.int8)
        board[4, 4], board[4, 5] = 1, -1
        baseline = CTeacher(depth=1, width=6, nodes=0, workers=1)
        expected, stats = baseline.policy_batch(board[None], [1])
        candidates = np.isfinite(stats['scores'][0]).sum()
        for budget, depth in [(1, 0), (int(candidates)+1, 1)]:
            policy, actual = CTeacher(depth=3, width=6, nodes=budget).policy_batch(board[None], [1])
            self.assertEqual(actual['depths'][0], depth)
            self.assertLessEqual(actual['nodes'][0], budget)
            np.testing.assert_allclose(actual['scores'], stats['scores'])
            np.testing.assert_allclose(policy, expected)
        boards = np.stack([board, -board, np.rot90(board)])
        before = boards.copy()
        single = CTeacher(depth=3, nodes=500, workers=1).policy_batch(boards, [1, -1, 1])
        multi = CTeacher(depth=3, nodes=500, workers=3).policy_batch(boards, [1, -1, 1])
        np.testing.assert_array_equal(single[0], multi[0])
        for key in single[1]:
            np.testing.assert_array_equal(single[1][key], multi[1][key])
        np.testing.assert_array_equal(boards, before)

    def test_temperature_legality_and_invalid_inputs(self):
        board = np.zeros((9, 9), np.int8)
        board[4, 4], board[4, 5], board[3, 4] = 1, -1, 1
        teacher = CTeacher(depth=2, nodes=1000)
        cold, _ = teacher.policy_batch(board[None], [-1], .01)
        hot, _ = teacher.policy_batch(board[None], [-1], 10000.)
        entropy = lambda p: -(p * np.log(np.maximum(p, 1e-30))).sum()
        self.assertGreater(entropy(hot), entropy(cold))
        self.assertFalse(hot[0, board.ravel() != 0].any())
        for temperature in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                teacher.policy_batch(board[None], [1], temperature)
        board[0, :5] = 1
        with self.assertRaises(ValueError):
            teacher.policy_batch(board[None], [-1])

    def test_soft_augmentation_and_loss_preserve_multiple_targets(self):
        states = torch.arange(25, dtype=torch.int8).reshape(1, 5, 5)
        targets = torch.zeros(1, 25)
        targets[0, 3], targets[0, 19] = .3, .7
        for seed in range(8):
            torch.manual_seed(seed)
            x, y = augment(states, targets)
            self.assertEqual(float(y[0, (x.flatten() == 3).nonzero()[0]]), float(targets[0, 3]))
            self.assertEqual(float(y[0, (x.flatten() == 19).nonzero()[0]]), float(targets[0, 19]))
            self.assertEqual(float(y.sum()), 1.)
        logits = torch.zeros(1, 25, requires_grad=True)
        F.cross_entropy(logits, targets).backward()
        self.assertLess(logits.grad[0, 3], 0)
        self.assertLess(logits.grad[0, 19], logits.grad[0, 3])
        self.assertGreater(logits.grad[0, 0], 0)


class RowPolicy(torch.nn.Module):
    def __init__(self, row):
        super().__init__()
        self.register_buffer('row', torch.tensor(row))

    def evaluate(self, states):
        logits = torch.full_like(states, -100, dtype=torch.float32)
        for i, state in enumerate(states):
            empty = (state[int(self.row)] == 0).nonzero().flatten()
            logits[i, int(self.row), empty[0]] = 100
        return logits, torch.zeros(len(states), device=states.device)

    def forward(self, states):
        return self.evaluate(states)[0]


class HistoryTests(unittest.TestCase):
    def test_only_current_actions_and_discount_include_opponent_plies(self):
        model = RowPolicy(0)
        pool = HistoryPool(2, 11)
        pool.add(model, 0)
        model.row.fill_(4)
        data = collect_history_games(model, pool, 2, 5, 'cpu', gamma=.5,
                                     history_probability=1, opening_moves=0)
        self.assertEqual(len(data['actions']), 9)
        self.assertEqual(data['environment_steps'], 18)
        self.assertTrue((data['actions'] >= 20).all())
        torch.testing.assert_close(data['outcomes'], torch.tensor([1, -1]))
        expected = data['outcomes'][data['game_ids']] * .5 ** (8-data['plies'])
        torch.testing.assert_close(data['returns'], expected)
        self.assertEqual(int(pool.snapshots[0]['model']['row']), 0)

    def test_frozen_weights_fifo_and_sampling_resume(self):
        model = ActorCritic(8, 1)
        pool = HistoryPool(2, 31)
        pool.add(model, 0)
        before = deepcopy(pool.snapshots[0]['model'])
        with torch.no_grad():
            next(model.parameters()).add_(1)
        for key, value in before.items():
            torch.testing.assert_close(value, pool.snapshots[0]['model'][key])
        pool.add(model, 1)
        pool.add(model, 2)
        self.assertEqual([x['update'] for x in pool.snapshots], [1, 2])
        self.assertTrue((pool.sample(4, 0) == -1).all())
        saved = deepcopy(pool.state_dict())
        expected = pool.sample(40, .75)
        restored = HistoryPool(2, 999)
        restored.load_state_dict(saved)
        torch.testing.assert_close(expected, restored.sample(40, .75))
        torch.testing.assert_close(expected[::2], expected[1::2])
        models = restored.opponents(model, torch.tensor([0, 1]))
        self.assertTrue(all(not p.requires_grad for i in (0, 1) for p in models[i].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_old_logs_values_legality_and_balanced_sides(self):
        torch.manual_seed(7)
        model = ActorCritic(8, 1)
        pool = HistoryPool(2, 7)
        pool.add(model, 0)
        data = collect_history_games(model, pool, 4, 5, 'cpu', game_offset=4)
        logits, values = model.evaluate(data['states'])
        _, logs, _ = policy_statistics(logits, data['states'], data['actions'])
        torch.testing.assert_close(logs, data['log_probs'])
        torch.testing.assert_close(values, data['values'])
        self.assertTrue((data['states'].flatten(1).gather(1, data['actions'][:, None]) == 0).all())
        self.assertEqual(data['environment_steps'], int(data['lengths'].sum())-8)
        torch.testing.assert_close(data['sides'], torch.tensor([1, -1, 1, -1]))
        torch.testing.assert_close(data['states'].flatten(1).sum(1) == 0, data['sample_sides'] == 1)

    @unittest.skipUnless(torch.cuda.is_available(), '需要可访问的 CUDA 设备')
    def test_cuda_history_collection(self):
        model = ActorCritic(8, 1).cuda()
        pool = HistoryPool(2, 7)
        pool.add(model, 0)
        data = collect_history_games(model, pool, 4, 9, 'cuda', precision='bf16')
        self.assertEqual(data['states'].device.type, 'cuda')
        self.assertTrue(torch.isfinite(data['log_probs']).all())
        self.assertTrue((data['states'].flatten(1).gather(1, data['actions'][:, None]) == 0).all())


class PipelineTests(unittest.TestCase):
    def test_distillation_training_bc_and_resume_with_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def run(name, expected_error=None, **parameters):
                cfg = dict(threads=1, seed=9, log_dir=str(root/name), **parameters)
                path = root / (name+'.yaml')
                write_yaml(path, cfg)
                process = subprocess.run([sys.executable, '-m', 'gomoku.train', '--config', str(path)],
                                         capture_output=True, text=True, timeout=90)
                if expected_error:
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn(expected_error, process.stderr)
                else:
                    self.assertEqual(process.returncode, 0, process.stdout+process.stderr)

            run('expert', command='expert', output=str(root/'data'), games=4, parallel_games=2,
                size=5, depth=2, width=6, nodes=100, workers=2, opening_moves=2,
                distillation_temperature=100.)
            dataset = ExpertDataset(root/'data')
            for states, targets, _ in dataset.batches(32):
                self.assertEqual(targets.ndim, 2)
                self.assertTrue((targets[states.flatten(1) != 0] == 0).all())
            run('imitate', command='imitate', data=str(root/'data'), output=str(root/'initial.pt'),
                channels=8, blocks=1, epochs=1, batch_size=16, device='cpu', deterministic=True,
                data_cache_mb=0)
            config = read_yaml(root/'initial.config.yaml')
            self.assertEqual(config['metadata']['policy_target'], 'distribution')
            for name, resume in [('imitate_full', None), ('imitate_resumed', str(root/'initial.pt'))]:
                run(name, command='imitate', data=str(root/'data'), output=str(root/(name+'.pt')),
                    channels=8, blocks=1, epochs=2, batch_size=16, device='cpu',
                    deterministic=True, resume=resume)
            full_imitation = torch.load(root/'imitate_full.pt', weights_only=True)
            resumed_imitation = torch.load(root/'imitate_resumed.pt', weights_only=True)
            for key, value in full_imitation['model'].items():
                torch.testing.assert_close(value, resumed_imitation['model'][key], rtol=0, atol=0)
            torch.testing.assert_close(full_imitation['rng_cpu'], resumed_imitation['rng_cpu'], rtol=0, atol=0)
            common = dict(command='selfplay', checkpoint=str(root/'initial.pt'), device='cpu',
                          deterministic=True, parallel_games=4, batch_size=32, ppo_epochs=2,
                          history_pool_size=2, history_interval=1, history_probability=.75,
                          history_inference='vmap', data_cache_mb=1,
                          bc_data=str(root/'data'), bc_batch_size=16, save_every=1)
            for name, updates, resume in [('full', 3, None), ('part', 1, None),
                                          ('resumed', 3, str(root/'part.pt'))]:
                run(name, **common, output=str(root/(name+'.pt')), updates=updates, resume=resume)
            expected = torch.load(root/'full.pt', weights_only=True)
            actual = torch.load(root/'resumed.pt', weights_only=True)

            def same(a, b):
                if isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                elif isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for key in a:
                        same(a[key], b[key])
                elif isinstance(a, list):
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b):
                        same(x, y)
                else:
                    self.assertEqual(a, b)

            for key in ('model', 'optimizer', 'history_pool', 'rng_cpu', 'env_steps', 'learner_steps'):
                same(expected[key], actual[key])
            run('bad_resume', **(common | {'history_interval': 2}), output=str(root/'bad.pt'),
                updates=4, resume=str(root/'full.pt'), expected_error='history_interval')
            run('bad_temperature', command='expert', output=str(root/'data'), games=6, parallel_games=2,
                size=5, depth=2, width=6, nodes=100, workers=2, opening_moves=2,
                distillation_temperature=200., resume=True, expected_error='distillation_temperature')
            manifest = read_yaml(root/'data/manifest.yaml')
            manifest['format_version'] = 1
            write_yaml(root/'data/manifest.yaml', manifest)
            with self.assertRaisesRegex(ValueError, '格式 2'):
                ExpertDataset(root/'data')
            run('old_data_resume', command='expert', output=str(root/'data'), games=6, resume=True,
                expected_error='旧单落点数据')


if __name__ == '__main__':
    unittest.main()

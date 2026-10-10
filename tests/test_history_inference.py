from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from gomoku.history import BatchedHistoryInference, HistoryPool, collect_history_games
from gomoku.model import ActorCritic
from gomoku.rollout import policy_statistics


torch.set_num_threads(1)


class HistoryInferenceTests(unittest.TestCase):
    def test_cached_models_are_reused_frozen_and_evicted_by_update(self):
        model = ActorCritic(8, 1)
        pool = HistoryPool(2, 17)
        pool.add(model, 0)
        pool.add(model, 1)
        first = pool.opponents(model, torch.tensor([0, 1]))
        with patch('gomoku.history.deepcopy', side_effect=AssertionError('recreated model')):
            second = pool.opponents(model, torch.tensor([0, 1]))
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        key = next(iter(pool.snapshots[0]['model']))
        self.assertEqual(pool.snapshots[0]['model'][key].data_ptr(), first[0].state_dict()[key].data_ptr())
        before = deepcopy(first[1].state_dict())
        with torch.no_grad():
            next(model.parameters()).add_(1)
        pool.add(model, 2)
        self.assertNotIn(0, pool._models)
        self.assertIs(pool.opponents(model, torch.tensor([0]))[0], first[1])
        for key, value in before.items():
            torch.testing.assert_close(value, first[1].state_dict()[key], rtol=0, atol=0)

    def test_batched_logits_match_independent_networks_and_both_color_groups(self):
        torch.manual_seed(19)
        model = ActorCritic(8, 1)
        pool = HistoryPool(4, 17)
        for update in range(4):
            with torch.no_grad():
                next(model.parameters()).add_(.1)
            pool.add(model, update)
        assignments = torch.tensor([-1, -1, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3])
        sides = torch.tensor([1, -1]*8)
        opponents = pool.opponents(model, assignments)
        batch = BatchedHistoryInference(opponents, assignments, sides, 'cpu')
        observation = torch.randint(-1, 2, (16, 9, 9), dtype=torch.int8)
        for phase in range(2):
            covered = batch.current[phase].tolist()
            for forward, params, buffers, matrices, masks, selections, destinations in batch.buckets:
                states = observation[matrices[phase]].masked_fill(~masks[phase][:, :, None, None], 0)
                logits = forward(params, buffers, states)
                for row in range(len(states)):
                    index = int(assignments[int(matrices[phase][row, 0])])
                    expected = opponents[index](states[row])
                    torch.testing.assert_close(logits[row], expected, rtol=1e-5, atol=1e-6)
                covered.extend(destinations[phase].tolist())
            self.assertEqual(sorted(covered), (sides != (1 if phase == 0 else -1)).nonzero().flatten().tolist())

    def test_vmap_rollout_records_only_live_learner_actions_and_correct_old_logs(self):
        for opening_moves in (0, 3):
            torch.manual_seed(9)
            model = ActorCritic(8, 1)
            pool = HistoryPool(3, 12)
            for update in range(3):
                pool.add(model, update)
            data = collect_history_games(model, pool, 8, 5, 'cpu', inference='vmap',
                                         opening_moves=opening_moves, gamma=.9,
                                         length_weight=.2, length_scale=25)
            ids, plies = data['game_ids'], data['plies']
            self.assertTrue((plies < data['lengths'][ids]).all())
            self.assertTrue((data['sample_sides'] == data['sides'][ids]).all())
            self.assertTrue((data['states'].flatten(1).gather(1, data['actions'][:, None]) == 0).all())
            logits, values = model.evaluate(data['states'])
            _, logs, _ = policy_statistics(logits, data['states'], data['actions'])
            torch.testing.assert_close(logs, data['log_probs'])
            torch.testing.assert_close(values, data['values'])
            expected = data['outcomes'][ids] * (1-.2*data['lengths'][ids]/25) * .9**(data['lengths'][ids]-1-plies)
            torch.testing.assert_close(data['returns'], expected)

    @unittest.skipUnless(torch.cuda.is_available(), '需要可访问的 CUDA 设备')
    def test_gpu_snapshots_stay_resident_across_batches_and_cpu_save_restore(self):
        model = ActorCritic(8, 1).cuda()
        pool = HistoryPool(3, 12)
        pool.add(model, 0)
        pool.add(model, 1)
        assignments = torch.tensor([0, 1])
        first = pool.opponents(model, assignments)
        pointers = {i: next(first[i].parameters()).data_ptr() for i in (0, 1)}
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('unexpected D2H')):
            second = pool.opponents(model, assignments)
            pool.add(model, 2)
        for i in (0, 1):
            self.assertEqual(next(second[i].parameters()).data_ptr(), pointers[i])
        saved = pool.state_dict()
        self.assertTrue(all(value.device.type == 'cpu' for entry in saved['snapshots'] for value in entry['model'].values()))
        self.assertTrue(all(value.device.type == 'cuda' for entry in pool.snapshots for value in entry['model'].values()))
        restored = HistoryPool(3, 1)
        restored.load_state_dict(saved)
        restored.opponents(model, assignments)
        self.assertTrue(all(value.device.type == 'cuda' for entry in restored.snapshots for value in entry['model'].values()))
        data = collect_history_games(model, restored, 8, 9, 'cuda', precision='bf16')
        self.assertTrue(torch.isfinite(data['log_probs']).all())


if __name__ == '__main__':
    unittest.main()

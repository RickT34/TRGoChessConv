from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gomoku.config import write_yaml
from gomoku.data import ExpertDataset
from gomoku.model import ActorCritic
from gomoku.train import move_batch


class TrainingCacheTests(unittest.TestCase):
    def dataset(self, root):
        parts = []
        for i in range(3):
            states = torch.zeros(100, 5, 5, dtype=torch.int8)
            states[:, 0, i] = 1
            targets = (states.flatten(1) == 0).float() / 24
            shard = {'states': states, 'policy_targets': targets,
                     'returns': torch.linspace(-1, 1, 100),
                     'game_ids': torch.arange(100)+100*i}
            name = f'part-{i}.pt'
            torch.save(shard, root/name)
            parts.append({'file': name, 'positions': 100})
        write_yaml(root/'manifest.yaml', {'format_version': 2, 'size': 5,
                                         'completed_games': 300, 'shards': parts})

    def test_cached_and_streamed_batches_have_identical_order_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            plain = ExpertDataset(root)
            cached = ExpertDataset(root, cache_mb=1)
            for split in ('train', 'validation'):
                for seed in (1, 2):
                    a = list(plain.batches(13, split, seed))
                    b = list(cached.batches(13, split, seed))
                    self.assertEqual(len(a), len(b))
                    for one, two in zip(a, b):
                        for x, y in zip(one, two):
                            torch.testing.assert_close(x, y, rtol=0, atol=0)
            with patch('gomoku.data.torch.load', side_effect=AssertionError('cache miss')):
                list(cached.batches(13, seed=3))

    def test_lru_eviction_and_oversized_shard_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            dataset = ExpertDataset(root, cache_mb=.03)
            dataset._shard(0)
            dataset._shard(1)
            dataset._shard(0)
            dataset._shard(2)
            self.assertEqual(list(dataset._cache), [0, 2])
            self.assertLessEqual(dataset.cache_bytes, dataset.cache_limit)
            no_fit = ExpertDataset(root, cache_mb=.001)
            list(no_fit.batches(10))
            self.assertEqual(no_fit.cache_bytes, 0)

    def test_policy_only_forward_skips_value_head_without_changing_logits(self):
        model = ActorCritic(8, 1)
        states = torch.zeros(2, 9, 9)
        logits, _ = model.evaluate(states)
        with patch.object(model.value_head, 'forward', side_effect=AssertionError('unused value head')):
            torch.testing.assert_close(model(states), logits, rtol=0, atol=0)
        torch.testing.assert_close(model(states, return_values=True)[0], logits, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), '需要可访问的 CUDA 设备')
    def test_gpu_cache_reuses_shards_and_never_pins_cuda_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root)
            dataset = ExpertDataset(root, cache_device='cuda', cache_mb=1)
            list(dataset.batches(16))
            pointers = {i: data['states'].data_ptr() for i, data in dataset._cache.items()}
            with patch('gomoku.data.torch.load', side_effect=AssertionError('cache miss')):
                for batch in dataset.batches(16):
                    self.assertTrue(all(t.device.type == 'cuda' for t in batch))
                    moved = move_batch(batch, 'cuda')
                    self.assertEqual([t.data_ptr() for t in batch], [t.data_ptr() for t in moved])
            self.assertEqual(pointers, {i: data['states'].data_ptr() for i, data in dataset._cache.items()})


if __name__ == '__main__':
    unittest.main()

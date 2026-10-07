import unittest

import torch

from gomoku.history import HistoryPool, collect_history_games
from gomoku.rewards import length_reward


torch.set_num_threads(1)


class TwoRows(torch.nn.Module):
    def evaluate(self, states):
        logits = torch.full_like(states, -100, dtype=torch.float32)
        for i, state in enumerate(states):
            row = 0 if state.sum() == 0 else -1
            col = (state[row] == 0).nonzero()[0, 0]
            logits[i, row, col] = 100
        return logits, torch.zeros(len(states), device=states.device)

    def forward(self, states):
        return self.evaluate(states)[0]


class LengthRewardTests(unittest.TestCase):
    def test_faster_wins_slower_losses_ordering_and_cap(self):
        lengths = torch.tensor([9, 30, 100, 361, 400])
        wins = length_reward(torch.ones(5), lengths, .2, 361)
        losses = length_reward(-torch.ones(5), lengths, .2, 361)
        self.assertTrue((wins[:3] > wins[1:4]).all())
        self.assertTrue((losses[:3] < losses[1:4]).all())
        self.assertTrue((wins > 0).all())
        self.assertTrue((losses < 0).all())
        self.assertFalse(length_reward(torch.zeros(5), lengths, .2, 361).any())
        self.assertEqual(wins[3], wins[4])

    def test_history_returns_include_length_reward_and_ply_discount(self):
        model = TwoRows()
        reward = 1-.2*9/25
        pool = HistoryPool(2, 11)
        pool.add(model, 0)
        history = collect_history_games(model, pool, 2, 5, 'cpu', gamma=.9,
                                        opening_moves=0, history_probability=1,
                                        length_weight=.2, length_scale=25)
        expected = history['outcomes'][history['game_ids']] * reward * .9**(8-history['plies'])
        torch.testing.assert_close(history['returns'], expected)


if __name__ == '__main__':
    unittest.main()

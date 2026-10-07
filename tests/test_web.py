"""验证 Web 棋局与真实模型推理的衔接，无需启动服务器。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gomoku.game import Gomoku
from gomoku.model import ActorCritic, save_policy
from gomoku.web import App, GameSession

torch.set_num_threads(1)


class WebGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "policy.pt"
        model = ActorCritic(4, 1)
        # 均匀策略便于确定贪心应手；测试仍经过真正的卷积网络。
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        save_policy(cls.path, model, 19, {"test": True})

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def session(self, **kwargs):
        return GameSession(self.path, "policy.pt", **kwargs)

    def test_human_black_response_probabilities_and_undo(self):
        game = self.session()
        self.assertFalse(game.history)
        game.move(180, 0)
        state = game.snapshot()
        self.assertEqual([move["player"] for move in state["history"]], [1, -1])
        self.assertEqual(state["board"][9][9], 1)
        self.assertEqual(state["board"][0][0], -1)
        self.assertEqual(state["probabilities"][180], 0)
        self.assertAlmostEqual(sum(state["probabilities"]), 1, places=5)
        self.assertTrue(state["can_undo"])
        game.undo(1)
        self.assertEqual(game.version, 2)
        self.assertFalse(game.env.board.any())
        self.assertEqual(game.history, [])

    def test_human_white_ai_opens_and_undo_keeps_opening(self):
        game = self.session(human=-1)
        self.assertEqual(game.history[0]["player"], 1)
        self.assertEqual(game.env.to_play, -1)
        game.move(180, 0)
        self.assertEqual(len(game.history), 3)
        game.undo(1)
        self.assertEqual(len(game.history), 1)
        self.assertEqual(game.env.to_play, -1)
        self.assertFalse(game.snapshot()["can_undo"])

    def test_illegal_and_stale_moves_do_not_change_board(self):
        game = self.session()
        game.move(180, 0)
        before = game.env.board.copy()
        for action, version in [(180, 1), (-1, 1), (361, 1), (True, 1), (20, 0)]:
            with self.assertRaises(ValueError):
                game.move(action, version)
            np.testing.assert_array_equal(game.env.board, before)
            self.assertEqual(game.version, 1)

    def test_terminal_human_win_has_no_ai_reply_and_can_undo(self):
        game = self.session()
        # AI 逐个占据第一行；人类在最后一行率先五连。
        for col in range(5):
            game.move(18 * 19 + col, game.version)
        self.assertTrue(game.env.done)
        self.assertEqual(game.env.winner, 1)
        self.assertEqual(len(game.history), 9)
        self.assertIsNone(game.snapshot()["probabilities"])
        with self.assertRaises(ValueError):
            game.move(100, game.version)
        game.undo(game.version)
        self.assertFalse(game.env.done)
        self.assertEqual(len(game.history), 8)

    def test_ai_win_stops_game(self):
        game = self.session()
        for action in [40, 80, 120, 160, 200]:
            game.move(action, game.version)
        self.assertEqual(game.env.winner, -1)
        self.assertTrue(game.env.done)

    def test_sample_seed_is_local_and_repeatable(self):
        one = self.session(human=-1, mode="sample", seed=12)
        two = self.session(human=-1, mode="sample", seed=12)
        self.assertEqual(one.history[0]["action"], two.history[0]["action"])
        action = int(one.env.legal_actions()[0])
        one.move(action, 0)
        torch.rand(100)
        two.move(action, 0)
        self.assertEqual([m["action"] for m in one.history], [m["action"] for m in two.history])

    def test_export_replays_exact_board_and_preserves_weights(self):
        game = self.session()
        before = {key: value.clone() for key, value in game.model.state_dict().items()}
        game.move(180, 0)
        record = json.loads(json.dumps(game.export(), allow_nan=False))
        replay = Gomoku(record["size"])
        for move in record["moves"]:
            self.assertEqual(move["player"], replay.to_play)
            replay.step(move["action"])
        np.testing.assert_array_equal(replay.board, game.env.board)
        self.assertEqual(len(record["model"]["sha256"]), 64)
        for key, value in game.model.state_dict().items():
            torch.testing.assert_close(before[key], value)

    def test_models_sessions_and_bad_paths(self):
        app = App(self.directory.name)
        self.assertEqual(app.models(), ["policy.pt"])
        a = app.request("/api/new", {"model": "policy.pt"})
        b = app.request("/api/new", {"model": "policy.pt", "human": -1})
        self.assertNotEqual(a["session"], b["session"])
        app.request("/api/move", {"session": a["session"], "action": 180, "version": 0})
        self.assertEqual(len(app.request("/api/state", {"session": b["session"]})["history"]), 1)
        for name in ["../policy.pt", str(self.path), "missing.pt"]:
            with self.assertRaises(ValueError):
                app.request("/api/new", {"model": name})
        with self.assertRaises(ValueError):
            app.request("/api/state", {"session": "missing"})


if __name__ == "__main__":
    unittest.main()

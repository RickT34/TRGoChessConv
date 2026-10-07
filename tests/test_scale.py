import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gomoku.batch_game import BatchGomoku
from gomoku.config import parse_config, write_yaml
from gomoku.game import Gomoku, winner
from gomoku.model import ActorCritic, load_policy, save_policy
from gomoku.rollout import policy_statistics, ppo_loss
from gomoku.teacher import CTeacher, PATTERNS, WEIGHTS


torch.set_num_threads(1)


def reference_pattern_score(board, player):
    """不使用 DFA，用 Python 字符串逐个匹配（包括重叠命中）。"""
    n = len(board)
    total = 0.
    for dr, dc in [(0,1),(1,0),(1,1),(1,-1)]:
        for row in range(n):
            for col in range(n):
                if 0 <= row-dr < n and 0 <= col-dc < n:
                    continue
                text, r, c = '', row, col
                while 0 <= r < n and 0 <= c < n:
                    text += '0' if board[r,c]==0 else '1' if board[r,c]==player else '2'
                    r, c = r+dr, c+dc
                for pattern, weight in zip(PATTERNS, WEIGHTS):
                    for side in [False, True]:
                        pattern2 = pattern.translate(str.maketrans('12','21')) if side else pattern
                        for variant in set([pattern2, pattern2[::-1]]):
                            count = sum(text.startswith(variant, i) for i in range(len(text)))
                            total += count * weight * (-2.079848 if side else 2.)
    return total


class TeacherTests(unittest.TestCase):
    def test_dfa_score_matches_original_g11_pattern_semantics(self):
        teacher = CTeacher(depth=2)
        rng = np.random.default_rng(15)
        for size in [5,15,19]:
            for player in [-1,1]:
                board = rng.choice([-1,0,0,0,1],size=(size,size)).astype(np.int8)
                np.testing.assert_allclose(teacher.score(board,player), reference_pattern_score(board,player), rtol=1e-12, atol=1e-5)

    def test_tactical_win_block_and_input_unchanged(self):
        boards = np.zeros((8,19,19),dtype=np.int8)
        players = np.array([1,-1]*4)
        for i, (dr,dc) in enumerate([(0,1),(1,0),(1,1),(1,-1)]*2):
            r,c = (18,0) if dr==0 else (0,18) if dc<=0 else (0,0)
            color = players[i] if i<4 else -players[i]
            for k in range(4): boards[i,r+k*dr,c+k*dc] = color
        before = boards.copy()
        actions, stats = CTeacher(depth=3,nodes=3000).choose_batch(boards,players)
        np.testing.assert_array_equal(boards, before)
        for i, action in enumerate(actions):
            boards[i].flat[action] = players[i] if i<4 else -players[i]
            self.assertEqual(winner(boards[i]), players[i] if i<4 else -players[i])
        self.assertTrue((stats['nodes']<=3000).all())

    def test_parallel_boards_match_serial_exactly(self):
        rng = np.random.default_rng(2)
        boards = np.zeros((12,19,19),dtype=np.int8)
        for board in boards:
            for k,a in enumerate(rng.choice(361,8,replace=False)):
                board.flat[a] = 1 if k%2==0 else -1
        before = boards.copy()
        single = CTeacher(depth=3,width=10,nodes=2000,workers=1).choose_batch(boards,np.ones(12))
        multi = CTeacher(depth=3,width=10,nodes=2000,workers=4).choose_batch(boards,np.ones(12))
        np.testing.assert_array_equal(single[0], multi[0])
        np.testing.assert_array_equal(single[1]['nodes'], multi[1]['nodes'])
        np.testing.assert_array_equal(single[1]['depths'], multi[1]['depths'])
        np.testing.assert_array_equal(boards,before)

    def test_depth_two_matches_exhaustive_minimax(self):
        rng = np.random.default_rng(11)
        teacher = CTeacher(depth=2,width=0,nodes=0,workers=1)
        checked = 0
        while checked<10:
            board = rng.choice([-1,1],(5,5)).astype(np.int8)
            board.flat[rng.choice(25,5,replace=False)] = 0
            if winner(board): continue
            player = 1 if checked%2 else -1
            values = {}
            for a in np.flatnonzero(board.ravel()==0):
                board.flat[a]=player
                if winner(board)==player:
                    values[a]=1e12-1
                else:
                    responses=[]
                    for b in np.flatnonzero(board.ravel()==0):
                        board.flat[b]=-player
                        responses.append(-1e12+2 if winner(board)==-player else teacher.score(board,player))
                        board.flat[b]=0
                    values[a]=min(responses)
                board.flat[a]=0
            chosen = teacher.choose(board,player)
            self.assertAlmostEqual(values[chosen],max(values.values()),places=4)
            checked+=1

    def test_invalid_and_terminal_rejected(self):
        teacher=CTeacher()
        board=np.zeros((19,19),dtype=np.int8)
        board[0,:5]=1
        with self.assertRaises(ValueError): teacher.choose(board,-1)
        with self.assertRaises(ValueError): teacher.choose(np.zeros((26,26)),1)
        with self.assertRaises(ValueError): teacher.choose(np.full((19,19),3),1)


class BatchTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), '需要可访问的 CUDA 设备')
    def test_cuda_board_updates_match_cpu(self):
        rng=np.random.default_rng(9)
        cpu,gpu=BatchGomoku(4,9),BatchGomoku(4,9,'cuda')
        for _ in range(81):
            actions=torch.tensor([int(rng.choice((board.flatten()==0).nonzero().flatten().numpy()))
                                  if not cpu.done[i] else 0 for i,board in enumerate(cpu.board)])
            cpu.step(actions,validate=True)
            gpu.step(actions.cuda(),validate=True)
            torch.testing.assert_close(gpu.board.cpu(),cpu.board)
            torch.testing.assert_close(gpu.winners.cpu(),cpu.winners)
            torch.testing.assert_close(gpu.done.cpu(),cpu.done)
            if cpu.done.all(): break

    def test_matches_scalar_games_through_terminal_and_freezes_done_slots(self):
        rng=np.random.default_rng(14)
        for size in [5,9,19]:
            batch=BatchGomoku(4,size)
            games=[Gomoku(size) for _ in range(4)]
            for _ in range(size**2):
                actions=[]
                for game in games:
                    action=int(rng.choice(game.legal_actions())) if not game.done else 0
                    actions.append(action)
                    if not game.done: game.step(action)
                batch.step(torch.tensor(actions),validate=True)
                for i,game in enumerate(games):
                    np.testing.assert_array_equal(batch.board[i].numpy(),game.board)
                    self.assertEqual(int(batch.winners[i]),game.winner)
                    self.assertEqual(bool(batch.done[i]),game.done)
                if batch.done.all(): break

    def test_edge_diagonal_and_overline(self):
        env=BatchGomoku(4,19)
        for i,(dr,dc) in enumerate([(0,1),(1,0),(1,1),(1,-1)]):
            r,c=(18,0) if dr==0 else (0,18) if dc<=0 else (0,0)
            for k in range(4): env.board[i,r+k*dr,c+k*dc]=1
        env.step(torch.tensor([18*19+4,4*19+18,4*19+4,4*19+14]))
        self.assertTrue((env.winners==1).all())


    def test_clipped_ppo_blocks_overlarge_positive_policy_step(self):
        logits=torch.tensor([[[1.3862944,0.]]],requires_grad=True)
        states=torch.zeros((1,1,2))
        loss,_=ppo_loss(logits,torch.zeros(1),states,torch.tensor([0]),
            torch.tensor([np.log(.4)],dtype=torch.float32),torch.ones(1),torch.zeros(1),
            value_coefficient=0,entropy_coefficient=0)
        loss.backward()
        torch.testing.assert_close(logits.grad,torch.zeros_like(logits))

    def test_residual_checkpoint_compatible_with_web(self):
        from gomoku.web import GameSession
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt'
            model=ActorCritic(8,2)
            save_policy(path,model,19,{})
            loaded,checkpoint=load_policy(path)
            self.assertIsInstance(loaded,ActorCritic)
            torch.testing.assert_close(loaded(torch.zeros((1,19,19))),model(torch.zeros((1,19,19))))
            session=GameSession(path,'model.pt',human=-1)
            self.assertEqual(len(session.history),1)


class ConfigTests(unittest.TestCase):
    def test_yaml_types_and_cli_override(self):
        parser=argparse.ArgumentParser()
        sub=parser.add_subparsers(dest='command',required=True)
        p=sub.add_parser('imitate')
        p.add_argument('--epochs',type=int,default=1)
        p.add_argument('--deterministic',action=argparse.BooleanOptionalAction,default=False)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.yaml'
            write_yaml(path,{'command':'imitate','epochs':20,'deterministic':True,'metadata':{'test':1}})
            args=parse_config(parser,['--config',str(path),'--epochs','3','--no-deterministic'])
            self.assertEqual(args.epochs,3)
            self.assertFalse(args.deterministic)
            write_yaml(path,{'command':'imitate','epohs':20})
            with self.assertRaises(SystemExit): parse_config(parser,['--config',str(path)])


if __name__=='__main__': unittest.main()

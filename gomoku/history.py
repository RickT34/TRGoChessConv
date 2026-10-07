"""单学习者 PPO 的冻结历史对手池；池与抽样 RNG 随训练断点保存。"""

from copy import deepcopy

import torch

from .batch_game import BatchGomoku
from .rollout import autocast, policy_statistics
from .rewards import length_reward
import numpy as np


def random_openings(env, moves, radius, seed, game_offset=0):
    """同一对棋局使用相同随机开局，学习者交换执棋方；不消耗全局 RNG。"""
    if not 0 <= moves <= 4 or radius < 1:
        raise ValueError('随机开局须为 0..4 手，半径至少为 1')
    for i in range(env.batch):
        rng = np.random.default_rng(np.random.SeedSequence([seed, (game_offset+i)//2]))
        for turn in range(moves):
            empty = (env.board[i].flatten() == 0).nonzero().flatten().numpy()
            row, col = np.divmod(empty, env.size)
            local = empty[(abs(row-env.size//2) <= radius) & (abs(col-env.size//2) <= radius)]
            action = int(rng.choice(local))
            env.board[i].flatten()[action] = 1 if turn % 2 == 0 else -1
        env.lengths[i] = moves
        env.players[i] = 1 if moves % 2 == 0 else -1


class HistoryPool:
    def __init__(self, capacity, seed):
        if capacity < 1:
            raise ValueError('历史池容量必须为正数')
        self.capacity = capacity
        self.rng = torch.Generator().manual_seed(seed)
        self.snapshots = []

    def add(self, model, update):
        if self.snapshots and update <= self.snapshots[-1]['update']:
            raise ValueError('历史快照 update 必须递增')
        weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        self.snapshots.append({'update': update, 'model': weights})
        self.snapshots = self.snapshots[-self.capacity:]

    def sample(self, count, probability):
        """每两盘使用同一对手并交换先后手；-1 代表本批冻结的当前模型。"""
        if count < 2 or count % 2 or not 0 <= probability <= 1:
            raise ValueError('历史池对局数须为正偶数，历史抽样概率须在 [0,1]')
        if not self.snapshots:
            raise ValueError('历史池为空')
        pairs = count // 2
        use_history = torch.rand(pairs, generator=self.rng) < probability
        indices = torch.randint(len(self.snapshots), (pairs,), generator=self.rng)
        return torch.where(use_history, indices, -1).repeat_interleave(2)

    def opponents(self, model, assignments):
        """只把本批抽中的历史权重放到模型设备，不为历史模型维护优化器。"""
        result = {-1: model}
        for index in assignments.unique().tolist():
            if index >= 0:
                opponent = deepcopy(model)
                opponent.load_state_dict(self.snapshots[index]['model'])
                result[index] = opponent.eval().requires_grad_(False)
        return result

    def state_dict(self):
        return {'capacity': self.capacity, 'rng': self.rng.get_state(), 'snapshots': self.snapshots}

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity or not 1 <= len(state['snapshots']) <= self.capacity:
            raise ValueError('历史池断点容量或快照数量不一致')
        updates = [entry['update'] for entry in state['snapshots']]
        if updates != sorted(set(updates)):
            raise ValueError('历史快照 update 必须严格递增')
        self.snapshots = [{'update': entry['update'],
                           'model': {k: v.detach().cpu().clone() for k, v in entry['model'].items()}}
                          for entry in state['snapshots']]
        self.rng.set_state(state['rng'].cpu())


@torch.no_grad()
def collect_history_games(model, pool, count, size, device, gamma=1., precision='fp32',
                          history_probability=.75, opening_moves=2, opening_radius=3,
                          seed=0, game_offset=0, length_weight=0., length_scale=361):
    assignments = pool.sample(count, history_probability)
    opponents = pool.opponents(model, assignments)
    env = BatchGomoku(count, size, device)
    if opening_moves:
        opening = BatchGomoku(count, size)
        random_openings(opening, opening_moves, opening_radius, seed, game_offset)
        for key in ('board', 'players', 'lengths'):
            getattr(env, key).copy_(getattr(opening, key))
    sides = torch.where((torch.arange(count, device=device)+game_offset) % 2 == 0, 1, -1)
    opponent_groups = {key: (assignments == key).to(device) for key in opponents}
    states, actions, logs, values, game_ids, plies = [], [], [], [], [], []
    for _ in range(size**2-opening_moves):
        if env.done.all():
            break
        learner = ((~env.done) & (env.players == sides)).nonzero().flatten()
        opponent_turn = (~env.done) & (env.players != sides)
        observation = env.observation()
        moves = torch.zeros(count, dtype=torch.long, device=device)
        if len(learner):
            x = observation[learner]
            with autocast(device, precision):
                logits, value = model.evaluate(x)
            chosen, old_log, _ = policy_statistics(logits, x)
            states.append(x)
            actions.append(chosen)
            logs.append(old_log)
            values.append(value.float())
            game_ids.append(learner)
            plies.append(env.lengths[learner].clone())
            moves[learner] = chosen
        for key, opponent in opponents.items():
            indices = (opponent_turn & opponent_groups[key]).nonzero().flatten()
            if len(indices):
                x = observation[indices]
                with autocast(device, precision):
                    logits = opponent(x)
                moves[indices] = policy_statistics(logits, x)[0]
        env.step(moves)
    if not env.done.all():
        raise RuntimeError('历史对手轨迹未完成')
    ids, times = torch.cat(game_ids), torch.cat(plies)
    outcomes = env.winners * sides
    rewards = length_reward(outcomes, env.lengths, length_weight, length_scale)
    returns = rewards[ids] * gamma ** (env.lengths[ids]-1-times)
    opponent_updates = torch.tensor([pool.snapshots[i]['update'] if i >= 0 else -1
                                    for i in assignments.tolist()])
    return {'states': torch.cat(states), 'actions': torch.cat(actions),
            'log_probs': torch.cat(logs), 'values': torch.cat(values), 'returns': returns,
            'game_ids': ids, 'plies': times, 'sample_sides': sides[ids], 'sides': sides,
            'lengths': env.lengths, 'winners': env.winners, 'outcomes': outcomes,
            'opponent_updates': opponent_updates,
            'environment_steps': int(env.lengths.sum())-count*opening_moves}

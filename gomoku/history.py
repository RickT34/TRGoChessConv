"""单学习者 PPO 的冻结历史对手池；池与抽样 RNG 随训练断点保存。"""

from copy import deepcopy
from itertools import chain

import torch
from torch.func import functional_call, stack_module_state, vmap

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
        self._models = {}

    def add(self, model, update):
        if self.snapshots and update <= self.snapshots[-1]['update']:
            raise ValueError('历史快照 update 必须递增')
        opponent = deepcopy(model).eval().requires_grad_(False)
        self._models[update] = opponent
        # state_dict 共享冻结模型的存储，不另外保留一份 CPU 或 GPU 权重。
        self.snapshots.append({'update': update, 'model': opponent.state_dict()})
        self.snapshots = self.snapshots[-self.capacity:]
        retained = {entry['update'] for entry in self.snapshots}
        self._models = {key: value for key, value in self._models.items() if key in retained}

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
        """冻结权重常驻学习者设备；恢复的 CPU 权重仅首次使用时迁移一次。"""
        device = next(chain(model.parameters(), model.buffers()), torch.empty(0)).device
        for entry in self.snapshots:
            entry['model'] = {key: value.to(device) for key, value in entry['model'].items()}
        result = {-1: model}
        for index in assignments.unique().tolist():
            if index >= 0:
                entry = self.snapshots[index]
                update = entry['update']
                if update not in self._models:
                    opponent = deepcopy(model)
                    opponent.load_state_dict(entry['model'], assign=True)
                    self._models[update] = opponent.eval().requires_grad_(False)
                elif next(chain(self._models[update].parameters(), self._models[update].buffers()),
                          torch.empty(0)).device != device:
                    self._models[update].to(device)
                    entry['model'] = self._models[update].state_dict()
                result[index] = self._models[update]
        return result

    def state_dict(self):
        # 仅保存断点时搬到 CPU；运行中的池不受序列化影响。
        snapshots = [{'update': entry['update'],
                      'model': {key: value.detach().cpu() for key, value in entry['model'].items()}}
                     for entry in self.snapshots]
        return {'capacity': self.capacity, 'rng': self.rng.get_state(), 'snapshots': snapshots}

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity or not 1 <= len(state['snapshots']) <= self.capacity:
            raise ValueError('历史池断点容量或快照数量不一致')
        updates = [entry['update'] for entry in state['snapshots']]
        if updates != sorted(set(updates)):
            raise ValueError('历史快照 update 必须严格递增')
        self.snapshots = [{'update': entry['update'],
                           'model': {k: v.detach().clone() for k, v in entry['model'].items()}}
                          for entry in state['snapshots']]
        self._models.clear()
        self.rng.set_state(state['rng'].cpu())


class BatchedHistoryInference:
    """按每个模型的棋局数分桶，vmap 合并推理；权重堆叠只做设备内复制。

    每对棋局同对手、交换先后手，因此两种轮次复用相同的一组堆叠权重。
    索引在每批开始时确定；热路径不为每个对手调用 nonzero/CPU 同步。
    """
    def __init__(self, opponents, assignments, sides, device):
        self.learners = [(sides == side).nonzero().flatten().to(device) for side in (1, -1)]
        self.current = [((sides != side) & (assignments == -1)).nonzero().flatten().to(device)
                        for side in (1, -1)]
        buckets = {}
        for key in sorted(opponents):
            if key < 0:
                continue
            indices = [((sides != side) & (assignments == key)).nonzero().flatten() for side in (1, -1)]
            width = max(len(index) for index in indices)
            if width:
                buckets.setdefault(1 << (width-1).bit_length(), []).append((opponents[key], indices))
        self.buckets = []
        for width, entries in buckets.items():
            parameters, buffers = stack_module_state([entry[0] for entry in entries])
            template = entries[0][0]

            def forward(parameters, buffers, states, template=template):
                return functional_call(template, (parameters, buffers), (states,))

            matrices, masks, selections, destinations = [], [], [], []
            for phase in range(2):
                matrix = torch.zeros((len(entries), width), dtype=torch.long)
                real = torch.zeros_like(matrix, dtype=torch.bool)
                for row, (_, indices) in enumerate(entries):
                    matrix[row, :len(indices[phase])] = indices[phase]
                    real[row, :len(indices[phase])] = True
                selections.append(real.flatten().nonzero().flatten().to(device))
                destinations.append(matrix[real].to(device))
                matrices.append(matrix.to(device))
                masks.append(real.to(device))
            self.buckets.append((vmap(forward), parameters, buffers,
                                 matrices, masks, selections, destinations))

    def choose(self, model, observation, phase, moves, device, precision):
        current = self.current[phase]
        if len(current):
            states = observation[current]
            with autocast(device, precision):
                logits = model(states)
            moves[current] = policy_statistics(logits, states)[0]
        for forward, parameters, buffers, matrices, masks, selections, destinations in self.buckets:
            states = observation[matrices[phase]].masked_fill(~masks[phase][:, :, None, None], 0)
            with autocast(device, precision):
                logits = forward(parameters, buffers, states)
            real_logits = logits.flatten(0, 1)[selections[phase]]
            real_states = states.flatten(0, 1)[selections[phase]]
            moves[destinations[phase]] = policy_statistics(real_logits, real_states)[0]


@torch.no_grad()
def collect_history_games(model, pool, count, size, device, gamma=1., precision='fp32',
                          history_probability=.75, opening_moves=2, opening_radius=3,
                          seed=0, game_offset=0, length_weight=0., length_scale=361,
                          inference='auto'):
    if inference not in ('auto', 'grouped', 'vmap'):
        raise ValueError('history_inference 须为 auto、grouped 或 vmap')
    assignments = pool.sample(count, history_probability)
    opponents = pool.opponents(model, assignments)
    env = BatchGomoku(count, size, device)
    if opening_moves:
        opening = BatchGomoku(count, size)
        random_openings(opening, opening_moves, opening_radius, seed, game_offset)
        for key in ('board', 'players', 'lengths'):
            getattr(env, key).copy_(getattr(opening, key))
    cpu_sides = torch.where((torch.arange(count)+game_offset) % 2 == 0, 1, -1)
    sides = cpu_sides.to(device)
    batched = (BatchedHistoryInference(opponents, assignments, cpu_sides, device)
               if inference == 'vmap' or (inference == 'auto' and str(device).startswith('cuda')) else None)
    opponent_groups = {} if batched else {key: (assignments == key).to(device) for key in opponents}
    states, actions, logs, values, game_ids, plies = [], [], [], [], [], []
    valid = []
    for turn in range(size**2-opening_moves):
        if env.done.all():
            break
        phase = (turn+opening_moves) % 2
        learner = (batched.learners[phase] if batched else
                   ((~env.done) & (env.players == sides)).nonzero().flatten())
        observation = env.observation()
        if batched:
            observation = observation.masked_fill(env.done[:, None, None], 0)
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
            valid.append(~env.done[learner])
            moves[learner] = chosen
        if batched:
            batched.choose(model, observation, phase, moves, device, precision)
        else:
            opponent_turn = (~env.done) & (env.players != sides)
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
    valid = torch.cat(valid)
    ids, times = torch.cat(game_ids)[valid], torch.cat(plies)[valid]
    outcomes = env.winners * sides
    rewards = length_reward(outcomes, env.lengths, length_weight, length_scale)
    returns = rewards[ids] * gamma ** (env.lengths[ids]-1-times)
    opponent_updates = torch.tensor([pool.snapshots[i]['update'] if i >= 0 else -1
                                    for i in assignments.tolist()])
    return {'states': torch.cat(states)[valid], 'actions': torch.cat(actions)[valid],
            'log_probs': torch.cat(logs)[valid], 'values': torch.cat(values)[valid], 'returns': returns,
            'game_ids': ids, 'plies': times, 'sample_sides': sides[ids], 'sides': sides,
            'lengths': env.lengths, 'winners': env.winners, 'outcomes': outcomes,
            'opponent_updates': opponent_updates,
            'environment_steps': int(env.lengths.sum())-count*opening_moves}

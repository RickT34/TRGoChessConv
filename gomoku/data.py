"""按多盘轨迹生成教师数据，分片落盘；模仿学习每次只读取一个分片。"""

import hashlib
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch

from .batch_game import BatchGomoku
from .config import read_yaml


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def generate_wave(first_game, count, args, teacher):
    env = BatchGomoku(count, args.size)
    rngs = [np.random.default_rng(np.random.SeedSequence([args.seed, first_game+i])) for i in range(count)]
    states, labels, policies, ids, players = [], [], [], [], []
    depths = []
    for turn in range(args.size**2):
        active = (~env.done).nonzero().flatten()
        if not len(active):
            break
        policy, stats = teacher.policy_batch(env.board[active].numpy(), env.players[active].numpy(),
                                             args.distillation_temperature)
        targets = policy.argmax(axis=1)
        states.append(env.observation()[active].clone())
        labels.append(torch.from_numpy(targets.astype(np.int64)))
        policies.append(torch.from_numpy(policy))
        ids.append(active + first_game)
        players.append(env.players[active].clone())
        depths.extend(stats['depths'].tolist())
        actions = torch.zeros(count, dtype=torch.long)
        for index, target in zip(active.tolist(), targets):
            board = env.board[index].numpy()
            action = int(target)
            if turn < args.opening_moves or rngs[index].random() < args.explore:
                empty = np.flatnonzero(board.ravel() == 0)
                if turn < args.opening_moves:
                    row, col = np.divmod(empty, args.size)
                    near_center = (abs(row-args.size//2) <= args.opening_radius) & (abs(col-args.size//2) <= args.opening_radius)
                    empty = empty[near_center]
                action = int(rngs[index].choice(empty))
            actions[index] = action
        env.step(actions)
    game_ids = torch.cat(ids)
    sample_players = torch.cat(players)
    returns = env.winners[game_ids-first_game].float() * sample_players
    policy_targets = torch.cat(policies)
    shard = {'states': torch.cat(states), 'actions': torch.cat(labels), 'returns': returns,
             'policy_targets': policy_targets, 'game_ids': game_ids, 'size': args.size,
             'search_depths': torch.tensor(depths, dtype=torch.int16), 'format_version': 2}
    stats = {'positions': len(game_ids), 'mean_completed_depth': float(np.mean(depths)),
             'static_fallback_fraction': float(np.mean(np.asarray(depths) == 0))}
    return shard, stats


class ExpertDataset:
    def __init__(self, directory, validation_every=10, cache_device='cpu', cache_mb=0):
        self.directory = Path(directory)
        if validation_every < 2:
            raise ValueError('validation_every 至少为 2')
        self.validation_every = validation_every
        if cache_mb < 0:
            raise ValueError('data_cache_mb 必须非负')
        self.cache_device = torch.device(cache_device)
        self.cache_limit = int(cache_mb * 2**20)
        self.cache_bytes = 0
        self._cache = OrderedDict()
        self.manifest = read_yaml(self.directory / 'manifest.yaml')
        if self.manifest.get('format_version') != 2:
            raise ValueError('仅支持格式 2 的教师分布数据，请重新运行 expert')
        self.size = self.manifest['size']
        self.samples = sum(part['positions'] for part in self.manifest['shards'])
        self.fingerprint = hashlib.sha256((self.directory / 'manifest.yaml').read_bytes()).hexdigest()
        if self.manifest['completed_games'] < 2:
            raise ValueError('至少生成 2 局教师数据，才能按整局拆分训练集和验证集')

    def _shard(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        source = torch.load(self.directory / self.manifest['shards'][index]['file'],
                            weights_only=True, mmap=True)
        targets = source['policy_targets']
        if (targets.shape != (len(source['states']), self.size**2)
                or not torch.isfinite(targets).all() or (targets < 0).any()
                or not torch.allclose(targets.sum(1), torch.ones(len(targets)), atol=1e-5)
                or (targets[source['states'].flatten(1) != 0] != 0).any()):
            raise ValueError('蒸馏分布须归一化、非负，且仅包含合法落点')
        validation = source['game_ids'].remainder(self.validation_every) == 0
        data = {key: source[key] for key in ('states', 'policy_targets', 'returns')}
        data['train'] = (~validation).nonzero().flatten()
        data['validation'] = validation.nonzero().flatten()
        size = sum(t.numel() * t.element_size() for t in data.values())
        data['bytes'] = size
        if size <= self.cache_limit:
            while self.cache_bytes + size > self.cache_limit:
                _, old = self._cache.popitem(last=False)
                self.cache_bytes -= old['bytes']
            for key in ('states', 'policy_targets', 'returns'):
                data[key] = data[key].to(self.cache_device)
            self._cache[index] = data
            self.cache_bytes += size
        # 超过预算的单个分片维持 CPU mmap，不一次性占满显存。
        return data

    def batches(self, batch_size, split='train', seed=0):
        generator = torch.Generator().manual_seed(seed)
        shards = self.manifest['shards']
        order = torch.randperm(len(shards), generator=generator).tolist() if split == 'train' else range(len(shards))
        for i in order:
            data = self._shard(i)
            # 相邻局面不能泄漏到验证集：第 0、10、20…局全部作为验证局。
            indices = data[split]
            if split == 'train':
                indices = indices[torch.randperm(len(indices), generator=generator)]
            for index in indices.split(batch_size):
                if len(index):
                    index = index.to(data['states'].device)
                    yield data['states'][index], data['policy_targets'][index], data['returns'][index]

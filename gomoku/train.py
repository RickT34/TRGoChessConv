"""教学训练入口：教师软标签 → 策略/价值蒸馏 → 历史对手池 PPO。"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .batch_game import BatchGomoku
from .data import ExpertDataset, generate_wave, write_json
from .model import ActorCritic, load_policy, save_policy
from .rollout import autocast, ppo_loss
from .teacher import CTeacher
from .history import HistoryPool, collect_history_games, random_openings
from .config import parse_config, read_yaml, write_yaml, reusable_config

from .metrics import Metrics
from .rewards import length_reward

LOGGER = None


def start_logging(args, start=0):
    global LOGGER
    LOGGER = Metrics(args, start)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment(states, actions):
    targets = actions.reshape_as(states)
    k = int(torch.randint(4, ()).item())
    states, targets = torch.rot90(states, k, (-2, -1)), torch.rot90(targets, k, (-2, -1))
    if torch.rand(()) < .5:
        states, targets = states.flip(-1), targets.flip(-1)
    return states, targets.flatten(1)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def config(args, **extra):
    result = {k: v for k, v in vars(args).items() if k != 'func'}
    return {**result, 'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
            'observation': 'board * current_player', 'rules': 'freestyle, at least five',
            'gpu': torch.cuda.get_device_name() if str(getattr(args, 'device', '')).startswith('cuda') else None,
            **extra}


def prepare_output(args, resolved):
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not getattr(args, 'resume', None):
        raise ValueError(f'{path} 已存在，请换 --output 或使用 --resume')
    write_yaml(path.with_suffix('.config.yaml'), reusable_config(args, resolved))


def save_training(args, model, size, resolved, optimizer, progress, env_steps=0, best=None, **extra):
    save_policy(args.output, model, size, resolved, optimizer=optimizer.state_dict(),
                stage=args.command, progress=progress, env_steps=env_steps, best=best,
                rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [], **extra)


def restore(args, checkpoint, optimizer, stage):
    if checkpoint.get('stage') != stage or 'optimizer' not in checkpoint:
        raise ValueError('该文件不是对应阶段的完整训练断点')
    keys = ['batch_size', 'lr', 'seed', 'precision', 'deterministic', 'weight_decay',
            'beta1', 'beta2', 'grad_clip', 'validation_every']
    keys += ['data', 'channels', 'blocks', 'value_coefficient'] if stage == 'imitate' else [
        'parallel_games', 'gamma', 'ppo_epochs', 'clip', 'value_coefficient',
        'entropy', 'target_kl', 'bc_data', 'bc_coefficient', 'bc_batch_size']
    for key in keys:
        if checkpoint['config'].get(key) != getattr(args, key):
            raise ValueError(f'续训参数 {key} 与断点不一致，请沿用原配置')
    if stage == 'selfplay':
        for key in ('history_pool_size', 'history_interval', 'history_probability',
                    'opening_moves', 'opening_radius', 'length_weight', 'length_scale'):
            if checkpoint['config'].get(key) != getattr(args, key):
                raise ValueError(f'续训参数 {key} 与断点不一致，请沿用原配置')
    optimizer.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['rng_cpu'].cpu())
    if args.device.startswith('cuda') and checkpoint.get('rng_cuda'):
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['rng_cuda']])
    return checkpoint['progress']


def expert(args):
    if args.size < 5 or args.size > 25:
        raise ValueError('棋盘边长须为 5..25')
    teacher = CTeacher(args.depth, args.width, args.nodes, args.workers)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.yaml'
    resolved = config(args, **teacher.config(), policy_target='root_score_softmax',
                      search_labels='full-window root candidates at a common completed depth; static fallback at depth 0')
    if manifest_path.exists():
        if not args.resume:
            raise ValueError('数据目录已有 manifest，请换目录或使用 --resume')
        manifest = read_yaml(manifest_path)
        if manifest.get('format_version', 1) != 2:
            raise ValueError('旧单落点数据不能混入蒸馏标签，请使用新数据目录')
        for key in ['size', 'seed', 'parallel_games', 'depth', 'width', 'nodes', 'explore', 'opening_moves', 'opening_radius', 'distillation_temperature']:
            if manifest['config'].get(key) != getattr(args, key):
                raise ValueError(f'数据续生成参数 {key} 与 manifest 不一致')
        if manifest['config']['source_sha256'] != teacher.config()['source_sha256']:
            raise ValueError('教师源码发生变化，请使用新数据目录')
    else:
        if args.resume:
            raise ValueError('数据目录没有可续生成的 manifest')
        manifest = {'format_version': 2, 'size': args.size, 'config': resolved,
                    'completed_games': 0, 'shards': []}
    start_logging(args, manifest['completed_games'])
    start = time.perf_counter()
    generated = 0
    manifest['config'] = resolved
    write_yaml(directory / 'config.yaml', reusable_config(args, resolved))
    for first in range(manifest['completed_games'], args.games, args.parallel_games):
        count = min(args.parallel_games, args.games-first)
        shard, stats = generate_wave(first, count, args, teacher)
        name = f"part-{len(manifest['shards']):05d}.pt"
        temporary = directory / (name + '.tmp')
        torch.save(shard, temporary)
        temporary.replace(directory / name)
        manifest['shards'].append({'file': name, 'positions': len(shard['actions']), 'games': count})
        manifest['completed_games'] = first + count
        write_yaml(manifest_path, manifest)
        generated += stats['positions']
        LOGGER.log(first+count, args.games,
                   positions=sum(part['positions'] for part in manifest['shards']),
                   positions_per_second=generated/(time.perf_counter()-start),
                   mean_completed_depth=stats['mean_completed_depth'],
                   static_fallback_fraction=stats['static_fallback_fraction'])


def move_batch(batch, device):
    if device.startswith('cuda'):
        return tuple(t.pin_memory().to(device, non_blocking=True) for t in batch)
    return tuple(t.to(device) for t in batch)


@torch.no_grad()
def validate(model, dataset, args):
    model.eval()
    sums = torch.zeros(5, device=args.device)
    for batch in dataset.batches(args.batch_size, 'validation'):
        states, targets, returns = move_batch(batch, args.device)
        with autocast(args.device, args.precision):
            logits, values = model.evaluate(states)
        flat = logits.float().masked_fill(states != 0, -1e9).flatten(1)
        best_actions = targets.argmax(1)
        target_entropy = -(targets * targets.clamp_min(1e-30).log()).sum()
        sums += torch.stack((F.cross_entropy(flat, targets, reduction='sum'),
                             (flat.argmax(1) == best_actions).sum(),
                             (values.float()-returns).square().sum(),
                             torch.tensor(len(states), device=args.device), target_entropy))
    loss, correct, value_error, total, entropy = sums.cpu().tolist()
    if not total:
        raise ValueError('验证集为空，请生成更多完整对局')
    return {'validation_loss': loss/total, 'validation_accuracy': correct/total,
            'validation_policy_kl': max(0., (loss-entropy)/total),
            'validation_value_mse': value_error/total}


def imitate(args):
    dataset = ExpertDataset(args.data, args.validation_every)
    checkpoint = None
    if args.resume:
        model, checkpoint = load_policy(args.resume, args.device)
    else:
        model = ActorCritic(args.channels, args.blocks).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1,args.beta2))
    resolved = config(args, size=dataset.size, dataset_sha256=dataset.fingerprint,
                      policy_target='distribution',
                      teacher_config=dataset.manifest['config'])
    start_epoch, seen, best = 0, 0, float('inf')
    if checkpoint:
        if checkpoint['config']['dataset_sha256'] != dataset.fingerprint:
            raise ValueError('数据集发生变化，请启动新的训练实验')
        start_epoch = restore(args, checkpoint, optimizer, 'imitate')
        seen = checkpoint['env_steps']
        best = checkpoint.get('best', best)
    prepare_output(args, resolved)
    start_logging(args, start_epoch)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = torch.zeros(2, device=args.device)
        for batch in dataset.batches(args.batch_size, seed=args.seed+epoch):
            states, targets, returns = move_batch(batch, args.device)
            states, targets = augment(states, targets)
            with autocast(args.device, args.precision):
                logits, values = model.evaluate(states)
            ce = F.cross_entropy(logits.float().masked_fill(states != 0, -1e9).flatten(1), targets)
            loss = ce + args.value_coefficient * (values.float()-returns).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            seen += len(states)
            totals += torch.stack((loss.detach()*len(states),
                                   torch.tensor(len(states), device=args.device)))
        loss_sum, total = totals.cpu().tolist()
        metrics = validate(model, dataset, args)
        if metrics['validation_loss'] < best:
            best = metrics['validation_loss']
            save_policy(Path(args.output).with_name(Path(args.output).stem+'-best.pt'),
                        model, dataset.size, resolved)
        LOGGER.log(epoch+1, args.epochs, train_loss=loss_sum/total,
                   validation_policy_kl=metrics['validation_policy_kl'],
                   validation_accuracy=metrics['validation_accuracy'],
                   validation_value_mse=metrics['validation_value_mse'])
        save_training(args, model, dataset.size, resolved, optimizer, epoch+1, env_steps=seen, best=best)


def selfplay(args):
    """只收集学习者的动作，用冻结历史对手训练单个 PPO 模型。"""
    model, checkpoint = load_policy(args.resume or args.checkpoint, args.device)
    if not isinstance(model, ActorCritic):
        raise ValueError('PPO 需要价值头，请先完成 imitate 蒸馏')
    dataset = ExpertDataset(args.bc_data, args.validation_every) if args.bc_data else None
    size = checkpoint['size']
    if dataset and dataset.size != size:
        raise ValueError('模仿数据和模型的棋盘尺寸不同')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1,args.beta2))
    start_update, env_steps, learner_steps = 0, 0, 0
    if args.resume:
        start_update = restore(args, checkpoint, optimizer, args.command)
        env_steps = checkpoint['env_steps']
        learner_steps = checkpoint.get('learner_steps', env_steps)
        if checkpoint['config'].get('dataset_sha256') != (dataset.fingerprint if dataset else None):
            raise ValueError('模仿数据发生变化，请启动新的训练实验')
    pool = HistoryPool(args.history_pool_size, args.seed)
    if args.resume:
        if 'history_pool' not in checkpoint:
            raise ValueError('训练断点缺少历史池，不能恢复历史对手训练')
        pool.load_state_dict(checkpoint['history_pool'])
    else:
        pool.add(model, 0)
    resolved = config(args, size=size, channels=model.channels, blocks=model.blocks,
                      dataset_sha256=dataset.fingerprint if dataset else None,
                      initial_checkpoint=args.checkpoint,
                      reward_formula='outcome * (1 - length_weight * min(total_plies / length_scale, 1))',
                      discount_unit='one ply; only learner actions enter PPO')
    prepare_output(args, resolved)
    start_logging(args, start_update)
    model.train()  # GroupNorm 无移动统计量，收集与优化使用相同计算。
    for update in range(start_update, args.updates):
        sync(args.device)
        start = time.perf_counter()
        rollout = collect_history_games(model, pool, args.parallel_games, size, args.device,
            args.gamma, args.precision, args.history_probability, args.opening_moves,
            args.opening_radius, args.seed, update*args.parallel_games,
            args.length_weight, args.length_scale)
        sync(args.device)
        count = len(rollout['actions'])
        batch_env_steps = rollout.get('environment_steps', count)
        env_steps += batch_env_steps
        learner_steps += count
        advantages = rollout['returns'] - rollout['values']
        advantages = (advantages-advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        teacher_batches = iter(dataset.batches(args.bc_batch_size, seed=args.seed+update)) if dataset and args.bc_coefficient else None
        sums = torch.zeros(6, device=args.device)
        batches, early_stop = 0, False
        for _ in range(args.ppo_epochs):
            for indices in torch.randperm(count, device=args.device).split(args.batch_size):
                states = rollout['states'][indices]
                with autocast(args.device, args.precision):
                    logits, values = model.evaluate(states)
                loss, metrics = ppo_loss(logits, values, states, rollout['actions'][indices],
                    rollout['log_probs'][indices], advantages[indices], rollout['returns'][indices],
                    args.clip, args.value_coefficient, args.entropy)
                if args.target_kl and metrics[3].item() > args.target_kl:
                    early_stop = True
                    break
                bc_loss = torch.zeros((), device=args.device)
                if teacher_batches is not None:
                    batch = next(teacher_batches, None)
                    if batch is None:
                        teacher_batches = iter(dataset.batches(args.bc_batch_size, seed=args.seed+update))
                        batch = next(teacher_batches)
                    x, y, _ = move_batch(batch, args.device)
                    x, y = augment(x, y)
                    with autocast(args.device, args.precision):
                        predictions = model(x)
                    bc_loss = F.cross_entropy(predictions.float().masked_fill(x != 0, -1e9).flatten(1), y)
                    loss = loss + args.bc_coefficient * bc_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                sums += torch.cat((metrics, bc_loss.detach()[None]))
                batches += 1
            if early_stop:
                break
        sync(args.device)
        total_time = time.perf_counter()-start
        stats = (sums / max(1, batches)).cpu().tolist()
        outcomes = rollout['outcomes'].float()
        scores = (outcomes + 1) / 2
        LOGGER.log(update+1, args.updates,
            win_rate=(outcomes == 1).float().mean().item(),
            draw_rate=(outcomes == 0).float().mean().item(),
            black_score_rate=scores[rollout['sides'] == 1].mean().item(),
            white_score_rate=scores[rollout['sides'] == -1].mean().item(),
            mean_length=rollout['lengths'].float().mean().item(),
            mean_reward=length_reward(outcomes, rollout['lengths'], args.length_weight, args.length_scale).mean().item(),
            policy_loss=stats[0], value_loss=stats[1], entropy=stats[2],
            approximate_kl=stats[3], clip_fraction=stats[4], bc_loss=stats[5],
            kl_early_stop=int(early_stop), steps_per_second=batch_env_steps/total_time)
        if (update+1) % args.history_interval == 0:
            pool.add(model, update+1)
        if (update+1) % args.save_every == 0 or update+1 == args.updates:
            save_training(args, model, size, resolved, optimizer, update+1, env_steps,
                          learner_steps=learner_steps,
                          history_pool=pool.state_dict())


def evaluate(args):
    model, checkpoint = load_policy(args.checkpoint, args.device)
    model.eval()
    teacher = CTeacher(args.depth, args.width, args.nodes, args.workers)
    resolved = config(args, size=checkpoint['size'], teacher_config=teacher.config())
    prepare_output(args, resolved)
    results = {color: dict(wins=0, losses=0, draws=0) for color in ['black', 'white']}
    for first in range(0, args.games, args.parallel_games):
        count = min(args.parallel_games, args.games-first)
        env = BatchGomoku(count, checkpoint['size'])
        sides = torch.tensor([1 if (first+i)%2 == 0 else -1 for i in range(count)])
        random_openings(env, args.opening_moves, args.opening_radius, args.seed, first)
        while not env.done.all():
            policy_ids = ((~env.done) & (env.players == sides)).nonzero().flatten()
            opponent_ids = ((~env.done) & (env.players != sides)).nonzero().flatten()
            actions = torch.zeros(count, dtype=torch.long)
            if len(policy_ids):
                with torch.no_grad(), autocast(args.device, args.precision):
                    states = env.observation()[policy_ids].to(args.device)
                    logits = model(states).float().masked_fill(states != 0, -1e9).flatten(1)
                    selected = torch.multinomial(logits.softmax(1), 1).flatten() if args.sample else logits.argmax(1)
                actions[policy_ids] = selected.cpu()
            if len(opponent_ids):
                chosen, _ = teacher.choose_batch(env.board[opponent_ids].numpy(), env.players[opponent_ids].numpy())
                actions[opponent_ids] = torch.from_numpy(chosen.astype(np.int64))
            env.step(actions)
        for i in range(count):
            side, win = int(sides[i]), int(env.winners[i])
            outcome = 'draws' if win==0 else 'wins' if win==side else 'losses'
            results['black' if side==1 else 'white'][outcome] += 1
    print(json.dumps(results, ensure_ascii=False), flush=True)
    write_json(args.output, {'results': results, 'config_file': str(Path(args.output).with_suffix('.config.yaml'))})


def positive(text):
    value = int(text)
    if value < 1: raise argparse.ArgumentTypeError('必须为正整数')
    return value


def nonnegative(text):
    value = int(text)
    if value < 0: raise argparse.ArgumentTypeError('必须为非负整数')
    return value


def probability(text):
    value = float(text)
    if not 0 <= value <= 1: raise argparse.ArgumentTypeError('必须在 [0,1] 内')
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    outputs = dict(expert='data/c_teacher_distill', imitate='runs/residual-distillation.pt',
                   selfplay='runs/residual-history.pt', evaluate='runs/evaluation.json')
    for name, function in [('expert', expert), ('imitate', imitate),
                           ('selfplay', selfplay), ('evaluate', evaluate)]:
        p = sub.add_parser(name)
        p.set_defaults(func=function)
        p.add_argument('--seed', type=nonnegative, default=42)
        p.add_argument('--threads', type=positive, default=1)
        p.add_argument('--output', default=outputs[name])
        if name != 'evaluate':
            p.add_argument('--log-dir', help='TensorBoard 事件目录')
            p.add_argument('--log-every', type=positive, default=20,
                           help='每多少批生成/epoch/update 打印摘要；首轮和末轮始终打印')
        if name != 'expert':
            p.add_argument('--device', default='auto')
            p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
            p.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=False)
        if name in ('expert', 'evaluate'):
            p.add_argument('--games', type=positive, default=1000 if name == 'expert' else 100)
            p.add_argument('--parallel-games', type=positive, default=32)
            p.add_argument('--workers', type=positive, default=8)
            p.add_argument('--depth', type=positive, default=4)
            p.add_argument('--width', type=nonnegative, default=16)
            p.add_argument('--nodes', type=nonnegative, default=20000)
        if name in ('expert', 'selfplay', 'evaluate'):
            p.add_argument('--opening-moves', type=nonnegative, default=2)
            p.add_argument('--opening-radius', type=positive, default=3)
        if name in ('imitate', 'selfplay'):
            p.add_argument('--resume', help='完整训练断点；epochs/updates 表示总目标')
            p.add_argument('--batch-size', type=positive, default=256)
            p.add_argument('--lr', type=float, default=5e-4 if name == 'imitate' else 3e-5)
            p.add_argument('--weight-decay', type=float, default=1e-4)
            p.add_argument('--beta1', type=probability, default=.9)
            p.add_argument('--beta2', type=probability, default=.999)
            p.add_argument('--grad-clip', type=float, default=1.)
            p.add_argument('--validation-every', type=positive, default=10,
                           help='每 N 局中留出一整局验证，避免相邻局面泄漏')
            p.add_argument('--value-coefficient', type=float, default=.25 if name == 'imitate' else .5)
        if name == 'expert':
            p.add_argument('--size', type=positive, default=19)
            p.add_argument('--explore', type=probability, default=.1)
            p.add_argument('--resume', action='store_true')
            p.add_argument('--distillation-temperature', type=float, default=100.)
        elif name == 'imitate':
            p.add_argument('--data', default='data/c_teacher_distill')
            p.add_argument('--epochs', type=positive, default=120)
            p.add_argument('--channels', type=positive, default=96)
            p.add_argument('--blocks', type=positive, default=20)
        elif name == 'selfplay':
            p.add_argument('--checkpoint', default='runs/residual-distillation-best.pt')
            p.add_argument('--updates', type=positive, default=1000)
            p.add_argument('--parallel-games', type=positive, default=256)
            p.add_argument('--ppo-epochs', type=positive, default=4)
            p.add_argument('--gamma', type=probability, default=.99)
            p.add_argument('--clip', type=probability, default=.2)
            p.add_argument('--entropy', type=float, default=.01)
            p.add_argument('--target-kl', type=float, default=.03)
            p.add_argument('--bc-data', default=None)
            p.add_argument('--bc-coefficient', type=float, default=.1)
            p.add_argument('--bc-batch-size', type=positive, default=256)
            p.add_argument('--save-every', type=positive, default=10)
            p.add_argument('--history-pool-size', type=positive, default=64)
            p.add_argument('--history-interval', type=positive, default=10)
            p.add_argument('--history-probability', type=probability, default=.85)
            p.add_argument('--length-weight', type=probability, default=.4)
            p.add_argument('--length-scale', type=positive, default=100)
        else:
            p.add_argument('--checkpoint', default='runs/residual-history.pt')
            p.add_argument('--sample', action='store_true', help='从策略采样；默认贪心落子')
    return parser


def main(argv=None):
    global LOGGER
    parser = build_parser()
    args = parse_config(parser, argv)
    if getattr(args, 'opening_moves', 0) > 4:
        parser.error('--opening-moves 最大为 4，保证随机开局尚未成五')
    if getattr(args, 'length_weight', 0) >= 1:
        parser.error('--length-weight 必须小于 1，确保赢 > 和 > 输')
    if args.command == 'expert' and (not np.isfinite(args.distillation_temperature) or args.distillation_temperature <= 0):
        parser.error('--distillation-temperature 须为有限正数')
    if args.command in ('selfplay', 'evaluate') and args.parallel_games % 2:
        parser.error('--parallel-games 须为偶数，以交换先后手')
    if args.command == 'evaluate' and args.games % 2:
        parser.error('--games 须为偶数，以交换先后手')
    for name in ('lr', 'grad_clip', 'entropy', 'value_coefficient', 'bc_coefficient', 'target_kl', 'weight_decay'):
        value = getattr(args, name, 1.)
        if not np.isfinite(value) or value < 0 or (name in ('lr', 'grad_clip') and value == 0):
            parser.error(f'{name} 须为有限数值，学习率和梯度裁剪须为正，其余系数非负')
    if getattr(args, 'validation_every', 10) < 2:
        parser.error('validation-every 至少为 2')
    if getattr(args, 'beta1', .9) >= 1 or getattr(args, 'beta2', .999) >= 1:
        parser.error('Adam beta 须小于 1')
    torch.set_num_threads(args.threads)
    if hasattr(args, 'device'):
        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            parser.error('该 Python 环境无法访问 CUDA，请检查解释器和 GPU 访问权限')
        if not args.device.startswith('cuda'):
            args.precision = 'fp32'
        if args.deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.set_float32_matmul_precision('high')
    seed_everything(args.seed)
    if args.command == 'evaluate':
        args.func(args)
        return
    if args.log_dir is None:
        args.log_dir = str(Path('runs/tensorboard') / Path(args.output).stem)
    # 检查输出后才初始化日志，防止误启动清空已有实验的记录。
    output = Path(args.output)
    existing = output / 'manifest.yaml' if args.command == 'expert' else output
    if existing.exists() and not args.resume:
        raise ValueError(f'{existing} 已存在，请换 --output 或使用 --resume')
    try:
        args.func(args)
    finally:
        if LOGGER is not None:
            LOGGER.close()
            LOGGER = None


if __name__ == '__main__':
    main()

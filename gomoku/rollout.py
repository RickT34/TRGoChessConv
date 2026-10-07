"""GPU 多轨迹收集与 PPO 损失；每批完整对局期间策略保持冻结。"""

from contextlib import nullcontext

import torch


def autocast(device, precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if str(device).startswith('cuda') and precision == 'bf16' else nullcontext()


def policy_statistics(logits, states, actions=None, generator=None):
    logits = logits.float().masked_fill(states != 0, -1e9).flatten(1)
    logs = logits.log_softmax(-1)
    probs = logs.exp()
    if actions is None:
        actions = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    return actions, logs.gather(1, actions[:, None]).squeeze(1), -(probs * logs).sum(1)


def ppo_loss(logits, values, states, actions, old_logs, advantages, returns,
             clip=.2, value_coefficient=.5, entropy_coefficient=.01):
    _, logs, entropy = policy_statistics(logits, states, actions)
    log_ratio = logs - old_logs
    ratio = log_ratio.exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1-clip, 1+clip) * advantages).mean()
    value = (values.float() - returns).square().mean()
    loss = policy + value_coefficient * value - entropy_coefficient * entropy.mean()
    with torch.no_grad():
        kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio-1).abs() > clip).float().mean()
    return loss, torch.stack((policy.detach(), value.detach(), entropy.mean().detach(), kl, clip_fraction))

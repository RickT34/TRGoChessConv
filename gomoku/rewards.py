"""有界的终局胜负／局长奖励，保持胜 > 和 > 负。"""


def length_reward(outcomes, lengths, weight=.2, scale=100):
    """R = outcome * (1 - weight * min(total_plies / scale, 1))."""
    if not 0 <= weight < 1 or scale <= 0:
        raise ValueError('length_weight 须在 [0,1)，length_scale 必须大于 0')
    fraction = (lengths.float() / scale).clamp(0, 1)
    return outcomes.float() * (1 - weight * fraction)

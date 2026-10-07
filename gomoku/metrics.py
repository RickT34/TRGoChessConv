"""只记录教学中使用的阶段指标；控制台节流，TensorBoard 保留每轮曲线。"""

import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Metrics:
    def __init__(self, args, start=0):
        self.stage = args.command
        self.every = args.log_every
        self.calls = 0
        output = Path(args.output)
        if self.stage == 'expert':
            output = output / 'generation'
        self.path = output.with_suffix('.metrics.jsonl')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 回到较早断点时，JSONL 和 TensorBoard 都丢弃该断点之后的记录。
        records = []
        if getattr(args, 'resume', None) and self.path.exists():
            records = [line for line in self.path.read_text().splitlines()
                       if json.loads(line)['step'] <= start]
        self.path.write_text(''.join(line + '\n' for line in records))
        self.writer = SummaryWriter(args.log_dir, purge_step=start+1 if start else None)

    def log(self, step, total, **metrics):
        record = {'stage': self.stage, 'step': step, **metrics}
        with self.path.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        for name, value in metrics.items():
            self.writer.add_scalar(f'{self.stage}/{name}', value, step)
        self.writer.flush()
        self.calls += 1
        if self.calls != 1 and self.calls % self.every and step != total:
            return
        if self.stage == 'expert':
            detail = (f"positions={metrics['positions']}  "
                      f"depth={metrics['mean_completed_depth']:.2f}  "
                      f"fallback={metrics['static_fallback_fraction']:.1%}  "
                      f"positions/s={metrics['positions_per_second']:.0f}")
        elif self.stage == 'imitate':
            detail = (f"loss={metrics['train_loss']:.4f}  "
                      f"val_KL={metrics['validation_policy_kl']:.4f}  "
                      f"val_acc={metrics['validation_accuracy']:.1%}  "
                      f"value_MSE={metrics['validation_value_mse']:.4f}")
        else:
            detail = (f"win={metrics['win_rate']:.1%}  draw={metrics['draw_rate']:.1%}  "
                      f"length={metrics['mean_length']:.1f}  "
                      f"policy={metrics['policy_loss']:.4f}  value={metrics['value_loss']:.4f}  "
                      f"entropy={metrics['entropy']:.3f}  KL={metrics['approximate_kl']:.4f}")
            if metrics['kl_early_stop']:
                detail += '  KL early stop'
        print(f'[{self.stage} {step}/{total}] {detail}', flush=True)

    def close(self):
        self.writer.close()

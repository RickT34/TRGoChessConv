"""验证精简日志、断点回退与调好配置的实际解析。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gomoku.config import parse_config, read_yaml
from gomoku.metrics import Metrics
from gomoku.train import build_parser


class MetricsTests(unittest.TestCase):
    def test_sparse_console_full_curves_and_resume_replaces_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(command='imitate', output=str(root/'model.pt'),
                                   log_dir=str(root/'tb'), log_every=3, resume=None)
            metrics = dict(train_loss=1., validation_policy_kl=.5,
                           validation_accuracy=.25, validation_value_mse=.1)
            stream = io.StringIO()
            with redirect_stdout(stream):
                logger = Metrics(args)
                for step in range(1, 6):
                    logger.log(step, 5, **metrics)
                logger.close()
            self.assertEqual(len(stream.getvalue().splitlines()), 3)  # first, every 3, final
            args.resume = args.output
            with redirect_stdout(io.StringIO()):
                logger = Metrics(args, start=2)
                logger.log(3, 3, **(metrics | {'train_loss': .2}))
                logger.close()
            records = [json.loads(line) for line in (root/'model.metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in records], [1, 2, 3])
            self.assertEqual(records[-1]['train_loss'], .2)
            events = EventAccumulator(str(root/'tb')).Reload()
            self.assertEqual(set(events.Tags()['scalars']), {f'imitate/{name}' for name in metrics})
            self.assertFalse(events.Tags()['tensors'])
            self.assertEqual([e.step for e in events.Scalars('imitate/train_loss')], [1, 2, 3])
            self.assertAlmostEqual(events.Scalars('imitate/train_loss')[-1].value, .2)

    def test_tuned_yaml_values_survive_parser(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('distill_expert', 'distill', 'selfplay_history'):
            path = root/'configs'/f'{name}.yaml'
            expected = read_yaml(path)
            actual = vars(parse_config(build_parser(), ['--config', str(path)]))
            for key, value in expected.items():
                self.assertEqual(actual[key], value, f'{name}: {key}')


if __name__ == '__main__':
    unittest.main()

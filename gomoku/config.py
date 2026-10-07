"""严格读取 YAML，命令行覆盖；保存的配置可以直接用于下一次运行。"""

import argparse
from pathlib import Path
import sys

import yaml


def read_yaml(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError('YAML 顶层必须是字段映射')
    return data


def write_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    temporary.replace(path)


def parse_config(parser, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config')
    known, rest = preliminary.parse_known_args(argv)
    if not known.config:
        return parser.parse_args(rest)
    data = read_yaml(known.config)
    command = data.pop('command', None)
    data.pop('metadata', None)
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    if command not in subparsers.choices:
        parser.error('YAML command 必须是 ' + ' / '.join(subparsers.choices))
    if rest and rest[0] in subparsers.choices:
        if rest.pop(0) != command:
            parser.error('命令行子命令与 YAML command 不一致')
    actions = {a.dest: a for a in subparsers.choices[command]._actions if a.option_strings}
    yaml_args = [command]
    for key, value in data.items():
        if key not in actions:
            parser.error(f'未知 YAML 参数：{key}')
        if value is None:
            continue
        action = actions[key]
        option = action.option_strings[0]
        if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
            if type(value) is not bool:
                parser.error(f'{key} 必须为 YAML 布尔值')
            if value:
                yaml_args.append(option)
            elif isinstance(action, argparse.BooleanOptionalAction):
                yaml_args.append('--no-' + option[2:])
        else:
            if isinstance(value,list) and action.nargs in ('+','*'):
                if any(isinstance(item,(dict,list,bool)) for item in value):
                    parser.error(f'{key} 的列表项须为数值或字符串')
                yaml_args.append(option)
                yaml_args.extend(str(item) for item in value)
                continue
            if isinstance(value, (dict, list, bool)):
                parser.error(f'{key} 须为数值或字符串')
            yaml_args.extend([option, str(value)])
    return parser.parse_args(yaml_args + rest)


def reusable_config(args, resolved):
    parameters = {k: v for k, v in vars(args).items() if k != 'func'}
    return {**parameters, 'metadata': {k: v for k, v in resolved.items() if k not in parameters}}

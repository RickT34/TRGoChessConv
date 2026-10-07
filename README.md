# TRGoChessConv

这是一个用五子棋讲解**模仿学习 + PPO 强化学习**的仓库。先让卷积神经网络学习 C 搜索程序的落子经验，再通过与历史版本对弈继续提升棋力，最终得到一个可以直接落子的策略模型。

规则为自由五子棋：连续五子及以上获胜，无禁手。默认 19×19。模型是 96 通道、20 个残差块的卷积网络，输出策略和价值。

[试玩网址(在浏览器中计算神经网络)](https://rickt34.github.io/misc/gomoku-rnn/)

## 技术路线

整个流程分为三个阶段，对应仓库中的三份训练配置：

1. **用 C 搜索教师生成数据**（`distill_expert.yaml`）。教师通过棋型评分和 Alpha-Beta 搜索评价候选落点，将评分转换为动作概率分布，保存对局中的局面、教师分布和最终胜负。
2. **模仿学习**（`distill.yaml`）。用教师的动作分布监督策略头，让网络学会选择落点；用对局胜负监督价值头，让网络学会评价局面。这一步也称为分布蒸馏，将搜索教师的经验学进网络参数。
3. **PPO 强化学习**（`selfplay_history.yaml`）。从蒸馏模型出发，与冻结的历史模型及当前版本对弈，根据胜负回报更新策略。采用简单的 PPO-Clip：以完整棋局回报减去价值估计作为优势，限制每次策略更新的幅度；同时保留少量模仿学习损失，帮助维持已经学到的落子经验。

历史对手池用于提供不同版本的对手，每对棋局交换先后手。训练过程中由网络直接选择动作，PPO 阶段不使用树搜索。具体损失函数、奖励和折扣的计算见 [训练说明](docs/training.md)。

## 达到的效果

相比于传统搜索方法，最直接的优势是极快的、常数级的决策时间。

**根据作者目前的实测，在已测试的对局设置下，训练后的模型在零搜索情况下已表现出较高棋力，超过了本项目使用的 C 语言搜索程序。** 这条路线展示了：先通过模仿学习获得基础能力，再用 PPO 对弈训练，可以得到超越初始搜索教师的直接落子策略。也印证了这样直觉：**一些需要深度搜索才能找到的较优策略，有可能本身就具有某种模式，并可以通过模式识别的方法直接找到。**

## 安装

需要 Python 3.11+、GCC 和 OpenMP（Linux 的 GCC 通常自带 OpenMP 支持）。首次使用教师时自动编译 `gomoku/native/teacher.c`，无需额外 C 项目。推荐在 Linux + NVIDIA CUDA 环境运行完整配置。

```bash
python -m venv .venv
source .venv/bin/activate
# 使用 GPU 时，先在该环境安装与显卡/驱动匹配的 PyTorch。
python -m pip install -e .
```

以下命令在仓库根目录执行。YAML 是实验参数的来源；命令行可覆盖同名参数，未知字段会报错。默认蒸馏和 PPO 使用 CUDA；小规模 CPU 检查可加 `--device cpu`，此时自动使用 FP32。

## 三阶段训练

```bash
python -m gomoku.train --config configs/distill_expert.yaml
python -m gomoku.train --config configs/distill.yaml
python -m gomoku.train --config configs/selfplay_history.yaml
```

| 阶段 | 调好的配置 | 输入 → 输出 | 核心参数 |
| --- | --- | --- | --- |
| 1. 教师数据 | `distill_expert.yaml` | C 搜索教师 → `data/c_teacher_distill/` | 5000 局，500 局并行，16 workers，深度 6，宽度 16，节点预算 30000，温度 100 |
| 2. 蒸馏 | `distill.yaml` | 教师软标签 → `runs/residual-distillation-best.pt` | 96 通道 × 20 块，120 epoch，batch 256，学习率 0.0005，价值系数 0.25 |
| 3. 历史池 PPO | `selfplay_history.yaml` | 最佳蒸馏模型 → `runs/residual-history.pt` | 256 局/更新，1000 更新，PPO 4 epoch，学习率 0.00003，历史概率 0.85，池容量 64，BC 系数 0.1 |

三份配置的数值沿用原实验；修正了历史池容量和抽样概率的旧注释。历史池中的权重保存在 CPU，每批抽中的对手会载入训练设备。完整配置较重，尤其是历史池及其训练断点；小显存机器可覆盖并行局数、batch 和池容量，但这属于新的实验配置。

蒸馏同时保存每个 epoch 的完整断点 `residual-distillation.pt`，以及按验证集策略交叉熵选出的 `residual-distillation-best.pt`。后者用于阶段 3 和对弈，不包含续训优化器状态。PPO 每 10 次更新及结束时保存完整断点。

数据、模型和日志都是本地生成物，不包含在源码中。

## 学习动态与日志

```bash
tensorboard --logdir runs/tensorboard
```

控制台每 `log_every` 轮打印一行摘要，首轮、末轮始终打印。这里一轮分别指一批教师对局、一个蒸馏 epoch、一次 PPO update；默认 20，可用 `--log-every 1` 查看每轮。TensorBoard 每轮记录核心曲线，不输出 minibatch 曲线、完整配置文本或每个历史对手的独立指标。

| 阶段 | TensorBoard 指标 | 主要用途 |
| --- | --- | --- |
| `expert`（4 项） | positions、positions_per_second、mean_completed_depth、static_fallback_fraction | 数据进度，以及节点预算是否让教师退回静态评分 |
| `imitate`（4 项） | train_loss、validation_policy_kl、validation_accuracy、validation_value_mse | 是否学到教师分布，以及价值预测误差 |
| `selfplay`（14 项） | win_rate、draw_rate、black_score_rate、white_score_rate、mean_length、mean_reward；policy_loss、value_loss、entropy、approximate_kl、clip_fraction、bc_loss、kl_early_stop、steps_per_second | 对局表现、PPO 稳定性和训练速度 |

横轴分别是累计教师局数、epoch、update。`score_rate` 为胜率 + 0.5 × 和棋率；`mean_reward` 为经过局长塑形、尚未折扣的学习者终局奖励。策略/价值损失等为本轮实际执行优化的 minibatch 均值；`kl_early_stop` 表示本轮因 KL 超限提前停止 PPO。

## 续训

```bash
# games / epochs / updates 均表示总目标，不是追加次数。
python -m gomoku.train --config configs/distill_expert.yaml --resume --games 6000
python -m gomoku.train --config runs/residual-distillation.config.yaml \
  --resume runs/residual-distillation.pt --epochs 160
python -m gomoku.train --config runs/residual-history.config.yaml \
  --resume runs/residual-history.pt --updates 2000
```

原子断点包含模型、优化器、随机数状态和训练进度；PPO 还包含冻结历史权重及对手抽样 RNG。更改训练数据、奖励、历史池等关键参数会拒绝续训。教师数据应在蒸馏开始前生成完毕；续生成数据改变指纹后，应启动新的蒸馏实验。`--checkpoint` 用于开启新 PPO 实验，`--resume` 用于恢复同一个实验。

## Web Server

```bash
python -m gomoku.web --models-dir runs --port 8000
```

打开 <http://127.0.0.1:8000>，选择模型和执棋方。支持最大概率落子、概率采样、神经网络 Minimax + Alpha-Beta 搜索，保留概率热力图、悔棋、棋谱导出和同一标签页刷新恢复。模型在一局内冻结，刷新模型列表后可开始新局加载新权重。

## 固定对手评估

保留一个简短的 C 教师评估入口，交换先后手、复用成对开局，帮助区分训练胜率与固定对手成绩。

```bash
python -m gomoku.train evaluate --checkpoint runs/residual-history.pt \
  --games 100 --parallel-games 20 --depth 6 --width 16 --nodes 30000 \
  --device cuda --output runs/evaluation.json
```

默认网络直接贪心落子，不使用 Web 的搜索增强；`--sample` 可切换采样。输出黑白双方的胜/负/和统计及旁边的配置。不同模型比较时应固定教师参数、种子、开局和落子方式。

## 代码阅读与测试

阅读顺序：`game.py` / `batch_game.py` → `teacher.py` / `data.py` → `model.py` → `train.py` → `history.py` / `rollout.py`。算法和回报公式见 [训练说明](docs/training.md)，教师迁移说明见 [native/README.md](gomoku/native/README.md)。

| 文件 | 职责 |
| --- | --- |
| `configs/distill_expert.yaml` | 教师数据生成配置：棋盘、对局数量、搜索预算、软标签温度等。 |
| `configs/distill.yaml` | 模仿学习配置：网络规模、学习率、batch、训练轮数等。 |
| `configs/selfplay_history.yaml` | PPO 配置：历史对手池、奖励、策略更新和模仿约束等。 |
| `gomoku/game.py` | 单盘五子棋规则、合法落子、胜负判断与当前玩家视角的观测。 |
| `gomoku/batch_game.py` | 用 PyTorch 张量批量维护棋局，可在 CPU 或 GPU 上并行推进多盘棋。 |
| `gomoku/native/teacher.c` | C 教师的棋型评分、Alpha-Beta 搜索及根候选评分；并行搜索不同棋局。 |
| `gomoku/teacher.py` | 自动编译 C 教师，通过 ctypes 提供批量选点和软标签接口。 |
| `gomoku/data.py` | 生成并分片保存教师数据，按整局划分训练/验证集，读取训练 batch。 |
| `gomoku/model.py` | 残差卷积策略/价值网络，合法动作分布，以及模型保存和加载。 |
| `gomoku/train.py` | 三阶段训练与固定教师评估入口，组织优化、验证、配置保存和断点续训。 |
| `gomoku/history.py` | 冻结历史模型池、对手抽样、成对开局，以及只记录学习者动作的轨迹采集。 |
| `gomoku/rollout.py` | 策略采样、动作 log probability、熵、PPO-Clip 损失和混合精度辅助函数。 |
| `gomoku/rewards.py` | 根据胜负和棋局长度计算终局奖励。 |
| `gomoku/config.py` | 读取 YAML、检查字段、应用命令行覆盖，保存可复用配置。 |
| `gomoku/metrics.py` | 输出精简的控制台摘要、TensorBoard 曲线和 JSONL 指标。 |
| `gomoku/web.py` | 本地 HTTP Server，提供页面、模型列表和对弈 API。 |
| `gomoku/session.py` | 维护每盘人机对弈的模型和状态，处理 AI 应手、悔棋及棋谱导出。 |
| `gomoku/neural_minimax.py` | Web 可选的神经网络搜索增强：策略选择候选，价值头评价叶节点。 |
| `gomoku/static/index.html` / `style.css` | 人机对弈页面的结构与样式。 |
| `gomoku/static/app.js` / `board.js` | 页面交互、服务端 API 调用，以及棋盘和概率热力图绘制。 |
| `tests/` | 棋盘、教师、训练、续训、日志和 Web 功能的自动化测试。 |
| `docs/training.md` / `docs/validation.md` | 算法与公式说明，以及精简版本的实际验证记录。 |
| `pyproject.toml` / `LICENSE` | Python 依赖与打包配置，以及 MIT 开源协议。 |

```bash
python -m unittest discover -s tests -v
```

## 许可

[MIT License](LICENSE)。C 教师保留 TRGoChessC 的棋型和评分权重来源说明。

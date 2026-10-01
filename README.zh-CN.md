# GenRec-Rerank

[语言选择](README.md) · [English](README.en.md)

GenRec-Rerank 研究基于三级语义 ID 的下一物品推荐，并提供离线重排分析工具。可运行的核心模型读取用户历史语义 ID，预测下一物品的 ID，再从给定的有效物品目录中选出完整 ID。仓库还包含候选列表、ID 冲突和重排结果的分析脚本。

## 方法

1. **语义 ID 输入。** 每件物品已经有一个三级 ID，例如 `<a_1><b_2><c_3>`。数据转换脚本从 CSV 中读取历史和目标物品。本项目使用已有 ID，不负责训练 ID 生成器。
2. **历史编码。** 分别嵌入每件物品的三个 ID 段并取平均，再加入位置编码。两层 Transformer 编码器产生用户历史表示。
3. **下一物品预测。** 三个输出层分别预测 ID 的三个段。训练时计算各段的交叉熵并取平均。
4. **有效物品筛选。** 预测时对给定目录中的每个完整 ID 累加三段得分，再选择得分最高的物品。这样输出一定属于给定目录。
5. **离线分析。** `analysis/` 中的脚本用于比较实验结果，检查物品级指标、ID 冲突、候选覆盖率和重排行为。这些脚本独立运行，训练命令不会自动调用它们。

模型与有效物品筛选代码位于 [`src/minionerec/model.py`](src/minionerec/model.py)。

## 数据流程

1. 准备包含 `history_item_sid` 和 `item_sid` 两列的 CSV。前者是非空的历史 ID 列表，后者是目标 ID。
2. 转换脚本检查每个 ID 是否有三个段，并将样本写成 JSONL。
3. 训练脚本从输入文件建立词表，对历史进行左侧填充，最多保留 100 件物品。
4. 模型训练后保存权重、词表和每轮损失。

JSONL 每行的格式如下：

```json
{"history": ["<a_1><b_1><c_2>", "<a_2><b_3><c_1>"], "target": "<a_3><b_2><c_4>"}
```

## 快速开始

需要 Python 3.9 或更新版本。在仓库根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

PYTHONPATH=src python -m minionerec.train examples/demo.jsonl \
  --epochs 1 --dim 16 --heads 2 --output runs/demo
```

示例数据是合成数据。运行后，`runs/demo/` 中会生成 `model.pt`、`vocabulary.json` 和 `metrics.json`。

使用含语义 ID 的 CSV 时，运行：

```bash
PYTHONPATH=src python -m minionerec.prepare data/train.csv data/train.jsonl
PYTHONPATH=src python -m minionerec.train \
  data/train.jsonl --epochs 5 --output runs/train
```

运行测试可使用 `PYTHONPATH=src python -m unittest discover -s tests -v`。查看单个离线脚本的参数可使用 `python analysis/compare_results.py --help`。

## 目录说明

| 路径 | 内容 |
| --- | --- |
| `src/minionerec/prepare.py` | 检查并转换语义 ID CSV 数据 |
| `src/minionerec/model.py` | 编码历史、预测 ID，并筛选有效候选 |
| `src/minionerec/train.py` | 训练并保存模型、词表和损失报告 |
| `analysis/` | 指标、ID 冲突和重排的独立分析脚本 |
| `results/` | 先前实验的部分统计结果 |
| `tests/` | 数据、模型、候选预测和训练测试 |

`results/` 中的统计结果来自先前实验，不能作为当前轻量模型的性能。本项目尚未实现语义 ID 构建、语言模型微调、强化学习和集成式重排训练。当前训练入口只输出损失；性能评估还需要独立测试集。

## 论文

[MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation](https://arxiv.org/abs/2510.24431)

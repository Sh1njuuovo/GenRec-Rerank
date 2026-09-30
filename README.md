# MiniOneRec 语义 ID 推荐复现

这个仓库提供一个独立编写的轻量实现。它将每个物品表示为三级语义 ID，读取用户历史语义 ID，并预测下一物品的三个 ID 段。候选预测只从有效的完整语义 ID 中选择。

## 运行

需要 Python 3.9 及以上版本和 PyTorch。先用合成数据验证训练流程。

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m minionerec.train examples/demo.jsonl --epochs 1 --dim 16 --heads 2
PYTHONPATH=src python -m unittest discover -s tests -v
```

已有包含 `history_item_sid` 与 `item_sid` 两列的 CSV 时，可以转换为训练数据。

```bash
PYTHONPATH=src python -m minionerec.prepare data/train.csv data/train.jsonl
PYTHONPATH=src python -m minionerec.train data/train.jsonl --output runs/train
```

JSONL 每行包含 `history` 列表和 `target` 字符串。语义 ID 格式为 `<a_1><b_2><c_3>`。0 作为填充值。`analysis/` 保存独立编写的实验分析工具，`results/` 保存之前实验的少量统计结果。

## 范围

这是一个小型语义 ID 序列模型，没有使用语言模型、强化学习或大规模训练。归档统计结果来自先前实验，不能视为当前代码的性能。当前训练入口只输出训练损失；正式推荐指标需要独立测试集。

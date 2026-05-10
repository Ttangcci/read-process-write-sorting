# Read-Process-Write Sorting Reproduction

这是一个 PyTorch 版本的最小复现项目，用于实现论文 *Order Matters: Sequence to Sequence for Sets* 中的 Read-Process-Write 思想，并在集合排序任务上和 Pointer Network baseline 做对比。

## 项目结构

```text
read-process-wrtie-sorting/
├── data.py              # 随机生成排序任务数据
├── modules.py           # Read / Process / Write 模块
├── model.py             # ReadProcessWriteModel 总模型
├── baseline.py          # Pointer Network baseline
├── train.py             # 单独训练 RPW 模型
├── evaluate.py          # 单独评估 RPW 模型
├── run_experiments.py   # 批量实验脚本
├── config.py            # 默认超参数
└── README.md
```

## 模型说明

`ReadProcessWriteModel` 包含三个阶段：

- `ReadModule`：对每个输入标量独立编码，得到 memory embeddings。
- `ProcessModule`：通过多步 process attention 反复读取整个集合，得到全局上下文。
- `WriteModule`：使用 LSTM decoder 输出排序后的输入下标序列。

`WriteModule` 中包含两类解码阶段 attention：

- `glimpse attention`：每个解码步先根据 decoder hidden state 读取一次 memory，用来细化 query。
- `pointer attention`：根据细化后的 query 给每个输入位置打分，输出对应的 pointer logits。

推理时会 mask 已经选择过的输入位置，避免重复输出同一个元素。训练时使用 teacher forcing，下一步 decoder 输入来自真实目标下标。

`baseline.py` 中实现了传统 Pointer Network baseline：

- LSTM encoder 读取输入序列；
- LSTM decoder 自回归输出；
- pointer attention 直接对 encoder memory 的每个位置打分。

## 环境要求

需要安装 PyTorch。可以先检查当前环境：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果没有安装 PyTorch，请先在对应 conda 环境中安装。

## 单独训练 RPW 模型

```powershell
python train.py
```

默认超参数位于 `config.py`：

```python
SET_SIZE = 5
TRAIN_SAMPLES = 20000
TEST_SAMPLES = 2000
BATCH_SIZE = 128
HIDDEN_DIM = 128
PROCESS_STEPS = 3
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
```

训练完成后会保存模型权重：

```text
rpw_sorting.pt
```

## 单独评估 RPW 模型

```powershell
python evaluate.py
```

评估会输出：

- `Test Exact Match Accuracy`：完整排序序列是否完全正确。
- `Valid Permutation Rate`：输出是否为合法排列，即没有重复选择或漏选。

## 运行批量实验

完整实验：

```powershell
python run_experiments.py
```

默认比较：

- 模型：`ReadProcessWriteModel` 和 `PointerNetwork`
- 集合大小：`N = 5, 10, 15`
- process steps：`P = 0, 1, 5, 10`

实验结果默认保存到：

```text
experiment_results.csv
```

该文件已加入 `.gitignore`，避免把本地实验输出误提交到仓库。

快速 smoke test：

```powershell
python run_experiments.py --epochs 1 --train-samples 200 --test-samples 100 --batch-size 32
```

只运行部分配置：

```powershell
python run_experiments.py --set-sizes 5 --process-steps 0,1 --epochs 3
```

指定输出文件：

```powershell
python run_experiments.py --output results_small.csv
```

## 结果字段

`run_experiments.py` 输出 CSV 字段包括：

- `model`
- `set_size`
- `process_steps`
- `epochs`
- `train_samples`
- `test_samples`
- `hidden_dim`
- `train_loss`
- `exact_match`
- `valid_permutation`
- `seconds`

## 当前实现状态

- [x] 合成排序数据集
- [x] Read-Process-Write 模型
- [x] process attention
- [x] glimpse attention
- [x] pointer attention
- [x] 推理阶段 mask 已选择位置
- [x] Pointer Network baseline
- [x] 单模型训练与评估脚本
- [x] 批量实验脚本
- [x] CSV 结果保存

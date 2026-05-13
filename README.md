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
├── plot_results.py      # 绘制实验图表
├── config.py            # 默认超参数
└── README.md
```

## 模型说明

`ReadProcessWriteModel` 包含三个阶段：

- `ReadModule`：对每个输入标量独立编码，得到 memory embeddings。
- `ProcessModule`：通过多步 process attention 反复读取整个集合，得到全局上下文。
- `WriteModule`：使用 LSTM decoder 输出排序后的输入下标序列。

`WriteModule` 支持 `glimpses=0` 和 `glimpses=1`：

- `glimpses=0`：decoder hidden state 直接用于 pointer attention。
- `glimpses=1`：每个解码步先执行一次 glimpse attention 读取 memory，再用细化后的 query 做 pointer attention。

推理时会 mask 已经选择过的输入位置，避免重复输出同一个元素。训练时使用 teacher forcing，下一步 decoder 输入来自真实目标下标。

`baseline.py` 中实现了传统 Pointer Network baseline，也支持 `glimpses=0/1`。

## 环境要求

需要安装 PyTorch。绘图脚本需要安装 `matplotlib`。

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import matplotlib; print(matplotlib.__version__)"
```

## 单独训练和评估 RPW

训练：

```powershell
python train.py
```

评估：

```powershell
python evaluate.py
```

评估会输出：

- `Test Exact Match Accuracy`：完整排序序列是否完全正确。
- `Valid Permutation Rate`：输出是否为合法排列，即没有重复选择或漏选。

## 批量实验

默认实验采用固定训练迭代次数，更接近论文中的 reported after 10000 training iterations：

```powershell
python run_experiments.py
```

默认配置：

- 训练长度：`train_set_size = 5`
- 测试长度：`test_set_sizes = 5,10,15`
- process steps：`P = 0,1,5,10`
- glimpses：`0,1`
- training iterations：`train_steps = 10000`
- 模型：`ReadProcessWriteModel` 和 `PointerNetwork`

手动指定完整配置：

```powershell
python run_experiments.py --train-set-size 5 --test-set-sizes 5,10,15 --process-steps 0,1,5,10 --glimpses 0,1 --train-steps 10000
```

快速 smoke test：

```powershell
python run_experiments.py --train-steps 10 --train-samples 200 --test-samples 100 --batch-size 32
```

指定输出文件：

```powershell
python run_experiments.py --output results_small.csv
```

实验结果默认保存到：

```text
experiment_results.csv
```

该文件已加入 `.gitignore`，避免把本地实验输出误提交到仓库。

## 结果字段

`run_experiments.py` 输出 CSV 字段包括：

- `model`
- `train_set_size`
- `test_set_size`
- `process_steps`
- `glimpses`
- `train_steps`
- `train_samples`
- `test_samples`
- `hidden_dim`
- `train_loss`
- `exact_match`
- `valid_permutation`
- `train_seconds`

## 绘制图表

```powershell
python plot_results.py
```

默认读取：

```text
experiment_results.csv
```

默认输出 PNG 图到：

```text
figures/
```

也可以指定输出格式：

```powershell
python plot_results.py --format svg
python plot_results.py --format pdf
```

## 当前实现状态

- [x] 合成排序数据集
- [x] Read-Process-Write 模型
- [x] process attention
- [x] `glimpses=0/1`
- [x] pointer attention
- [x] 推理阶段 mask 已选择位置
- [x] Pointer Network baseline
- [x] 固定 training iterations 的 out-of-sample 批量实验脚本
- [x] CSV 结果保存
- [x] matplotlib 图表脚本

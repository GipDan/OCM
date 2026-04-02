# Scripts Usage

## 目录说明

当前脚本目录已经拆成了薄入口 + 可复用模块：

- `benchmark_real_records.py`
  真实算子采样与写入 `records` 的命令行入口。
- `evaluate_real_train_test.py`
  对真实采样数据做 train/test 划分评估的命令行入口。
- `real_bench/benchmark_ops.py`
  算子注册表、case 配置、builder 逻辑。后续新增算子主要改这里。
- `real_bench/benchmark_cli.py`
  benchmark 的 CLI 参数与主流程。
- `real_bench/evaluation.py`
  评估、分组、训练测试拆分逻辑。
- `real_bench/common.py`
  公共常量、数据库写入、样本元信息、benchmark 公共函数。

## 环境选择

这套脚本依赖两个远端 conda 环境：

- `pytorch`
  用来跑真实 benchmark 采样，因为这里有 CUDA 和 `torch`。
- `ocm`
  用来跑评估、训练、推理相关逻辑，因为这里有 `xgboost` 和项目依赖。

建议统一在远端 `/home/dkw/OCM` 下执行。

## 1. 记录真实数据

先看当前支持哪些算子：

```bash
cd /home/dkw/OCM
conda run -n pytorch python scripts/benchmark_real_records.py --list-ops
```

只测单个算子并先做 dry-run：

```bash
conda run -n pytorch python scripts/benchmark_real_records.py \
  --op matmul_row_major_fp32 \
  --limit-per-op 5 \
  --warmup 5 \
  --repeats 10 \
  --dry-run
```

真正写入数据库：

```bash
conda run -n pytorch python scripts/benchmark_real_records.py \
  --op matmul_row_major_fp32 \
  --limit-per-op 20 \
  --warmup 10 \
  --repeats 20
```

一次写入多个算子：

```bash
conda run -n pytorch python scripts/benchmark_real_records.py \
  --op conv2d_nchw_fp32 \
  --op conv2d_nhwc_fp16 \
  --op matmul_row_major_fp16 \
  --limit-per-op 20
```

说明：

- `--op` 支持重复传入，也支持逗号分隔。
- `--limit-per-op` 控制每个算子最多采多少条 case。现在常用大算子已经预置到约 20 条。
- `--dry-run` 只测量不写库，适合先检查波动和耗时。
- 写入时会自动跳过数据库里已经存在的同语义样本，不会重复灌同一条记录。
- 新样本会在 `params["benchmark_meta"]` 中写入：
  - `op_key`
  - `sample_id`
  - `sample_label`
  - `case_note`
  - `stats`

数据库默认路径是：

```text
data/ocm.sqlite3
```

## 2. 如何新增算子

新增算子时，主要改 `scripts/real_bench/benchmark_ops.py`：

1. 新增一个 `make_xxx_case(...)`
   负责构造输入张量、`params`、运行函数和输出 shape。
2. 新增该算子的配置列表
   一般给 10 到 20 条具有代表性的 shape / layout / dtype / stride 组合。
3. 在 `AVAILABLE_SPECS` 中注册
   指定：
   - `key`
   - `aliases`
   - `builder`
   - `configs`

注册完成后，可以立刻用下面命令检查：

```bash
conda run -n pytorch python scripts/benchmark_real_records.py --list-ops
```

## 3. 评估真实样本

这个脚本会：

- 从 `records` 中筛出 `benchmark_meta.source == "real_pytorch_cuda_event"` 的样本
- 按 `(op_name, device, feature_order_key)` 分组
- 对每组做 train/test 划分
- 输出每组指标和整体指标
- 可选地把报告写成 JSON

运行方式：

```bash
cd /home/dkw/OCM
conda run -n ocm python scripts/evaluate_real_train_test.py
```

指定输出报告路径：

```bash
conda run -n ocm python scripts/evaluate_real_train_test.py \
  --report-path reports/real_train_test_report.json
```

如果想在评估后顺便把模型写进数据库：

```bash
conda run -n ocm python scripts/evaluate_real_train_test.py --store-models
```

常用参数：

- `--device`
  默认是 `NVIDIA_A100_80GB_PCIe`
- `--seed`
  控制 train/test 随机划分
- `--store-models`
  评估完成后对可评估组用全量样本训练并写入 `models`

## 4. 直接训练模型

如果你不想先跑 `evaluate_real_train_test.py`，也可以直接用项目 API 训练。

示例：

```bash
cd /home/dkw/OCM
conda run -n ocm python - <<'PY'
from ocm.database import get_connection, init_db
from ocm.train import fit_and_store_model

conn = get_connection("data/ocm.sqlite3")
init_db(conn)

ok, msg = fit_and_store_model(
    conn,
    op_name="nn::matmul_row_major_fp32",
    device="NVIDIA_A100_80GB_PCIe",
)
print(ok, msg)
conn.close()
PY
```

如果你想只训练某个 `feature_order_key` 对应的一组样本，也可以加：

```python
feature_order_key='["K","M","N","is_contiguous","memory_stride_0","memory_stride_1"]'
```

## 5. 推理预测

训练完成后，可以直接从数据库加载模型做推理：

```bash
cd /home/dkw/OCM
conda run -n ocm python - <<'PY'
from ocm.database import get_connection
from ocm.inference import predict_latency

conn = get_connection("data/ocm.sqlite3")

params = {
    "M": 1024,
    "N": 1024,
    "K": 512,
    "is_contiguous": True,
    "memory_stride": [512, 1],
}

pred = predict_latency(
    conn,
    op_name="nn::matmul_row_major_fp32",
    device="NVIDIA_A100_80GB_PCIe",
    params=params,
)
print("predicted_latency_ms =", pred)
conn.close()
PY
```

如果一个 `(op_name, device)` 下有多个模型，而你又不指定 `feature_order_key`，`predict_latency(...)` 可能会返回 `None`。这时要显式传入目标模型的 `feature_order_key`。

## 6. 推荐工作流

推荐按下面顺序做：

1. 在 `pytorch` 环境里先 `--dry-run` 检查算子是否稳定。
2. 确认没问题后正式写入 `records`。
3. 在 `ocm` 环境里跑 `evaluate_real_train_test.py` 看泛化效果。
4. 需要落库模型时，加 `--store-models`，或者直接调用 `fit_and_store_model(...)`。
5. 在业务代码里通过 `predict_latency(...)` 做推理。

## 7. 常见问题

### 为什么 benchmark 和 evaluate 不能在同一个环境里跑？

因为 benchmark 依赖 CUDA + `torch`，而 evaluate / train / inference 依赖 `xgboost` 和项目 Python 依赖。当前远端环境里：

- `pytorch` 适合采样
- `ocm` 适合评估、训练、推理

### 为什么 `records.id` 看起来不是连续按算子排的？

`records.id` 是数据库全局自增主键，不建议重排。现在脚本会给每条样本写：

- `benchmark_meta.sample_id`
- `benchmark_meta.sample_label`

所以更推荐用 `sample_id` 和 `sample_label` 理解同一个算子内部的样本顺序。

### 为什么有些 case 没写进去？

脚本会自动过滤波动太大的样本。如果某条 case 的 `cv` 超过阈值，即使重测后仍不稳定，也会跳过，不写入数据库。

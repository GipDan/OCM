# Scripts Usage

## 目录说明

当前脚本目录已经拆成了薄入口 + 可复用模块：

- `benchmark_real_records.py`
  真实算子采样与写入 `records` 的命令行入口。
- `evaluate_real_train_test.py`
  对真实采样数据做 train/test 划分评估的命令行入口。
- `train_real_operator.py`
  对单个算子做模型训练并写入 `models`。
- `predict_real_latency.py`
  对单个算子做延迟预测，可直接读取某条 record 或手动传 params。
- `real_bench/benchmark_ops.py`
  算子注册表、case 配置、builder 逻辑。后续新增算子主要改这里。
- `real_bench/benchmark_cli.py`
  benchmark 的 CLI 参数与主流程。
- `real_bench/evaluation.py`
  评估、分组、训练测试拆分逻辑。
- `real_bench/model_cli.py`
  单算子训练、模型列表、预测 CLI 逻辑。
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

如果你只是想把某个算子的样本补齐到目标数量，推荐直接用：

```bash
conda run -n pytorch python scripts/benchmark_real_records.py \
  --op matmul_row_major_fp32 \
  --top-up-to 20 \
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
- `--top-up-to` 会先看数据库当前已有多少条 `sample_id`，只补差额，更适合持续补样本。
- 默认会优先跳过当前库里已存在的 `sample_id`；如果你确实想强制重测，可加 `--rerun-existing`。
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

只看单个算子的评估：

```bash
conda run -n ocm python scripts/evaluate_real_train_test.py \
  --op matmul_row_major_fp32 \
  --report-path reports/matmul_fp32_eval.json
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

## 4. 单算子训练

推荐直接用训练入口，不用再手写 Python 片段。

先看这个算子有哪些样本分组：

```bash
cd /home/dkw/OCM
conda run -n ocm python scripts/train_real_operator.py \
  --op matmul_row_major_fp32 \
  --list-groups
```

训练该算子的全部分组并写入 `models`：

```bash
conda run -n ocm python scripts/train_real_operator.py \
  --op matmul_row_major_fp32 \
  --report-path reports/train_matmul_fp32.json
```

如果你只想训练某一个 `feature_order_key`：

```bash
conda run -n ocm python scripts/train_real_operator.py \
  --op matmul_row_major_fp32 \
  --feature-order-key '["K","M","N","is_contiguous","memory_stride_0","memory_stride_1"]'
```

## 5. 直接训练模型

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

## 6. 推理预测

训练完成后，可以直接从数据库加载模型做推理：

推荐直接用预测入口：

```bash
cd /home/dkw/OCM
conda run -n ocm python scripts/predict_real_latency.py \
  --op matmul_row_major_fp32 \
  --params-json '{"M":1024,"N":1024,"K":512,"is_contiguous":true,"memory_stride":[512,1]}'
```

如果你想直接拿某条已有 record 做预测，并同时对比真实值：

```bash
conda run -n ocm python scripts/predict_real_latency.py \
  --record-id 71
```

如果你只想看当前这个算子有哪些模型：

```bash
conda run -n ocm python scripts/predict_real_latency.py \
  --op matmul_row_major_fp32 \
  --list-models
```

新的预测逻辑会优先根据 `params` 自动推断最合适的 `feature_order_key`，所以同一个算子下有多个模型时，通常不需要你手工指定。

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

## 7. 推荐工作流

推荐按下面顺序做：

1. 在 `pytorch` 环境里先 `--dry-run` 检查当前算子是否稳定。
2. 用 `--top-up-to 20` 这类方式把现有算子样本补齐。
3. 在 `ocm` 环境里跑 `evaluate_real_train_test.py --op ...` 看单算子泛化效果。
4. 用 `train_real_operator.py` 把该算子的模型写入数据库。
5. 用 `predict_real_latency.py` 直接做单算子预测或 record 回放预测。

## 8. 常见问题

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

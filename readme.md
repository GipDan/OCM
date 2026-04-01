# ⚡ 算子 Cost Model 数据库系统架构说明书

## 一、 系统定位与核心设计理念
本系统旨在为深度学习编译器（如底层算子调度器）提供高精度的算子耗时预测。
* **零文件架构 (Zero-File Architecture)**：抛弃传统的 `.onnx` 或 `.pkl` 模型文件存储，将模型权重和结构完全序列化为文本，实现“单体数据库走天下”。
* **硬件状态感知 (Hardware-Aware)**：不仅关注算子的算法参数，还深度接入底层张量的物理内存排布状态。
* **极简技术栈**：使用 `SQLite` 作为元数据与模型载体，`XGBoost` 作为拟合引擎，`Streamlit` 提供交互式全生命周期管理。

---

## 二、 算子命名规范 (Naming Conventions)
为了确保数据库中 `op_name` 的全局唯一性并承载足够的上下文，系统采用**多级命名空间与后缀修饰**的工业级规范。

格式标准：`[命名空间]::[基础算子名]_[数据排布]_[数据精度]_[特殊属性(可选)]`

* **命名空间 (Namespace)**: `nn` (神经网络), `math` (数学), `vision` (视觉), `custom` (自定义)。
* **基础算子名 (Base Name)**: 如 `conv2d`, `matmul`, `reduce_sum`。
* **数据排布 (Layout)**: 决定计算逻辑，如 `nchw`, `nhwc`, `row_major`。
* **数据精度 (Precision)**: 决定硬件指令集，如 `fp32`, `fp16`, `int8`。
* **特殊属性 (Attributes)**: 如 `_relu` (算子融合), `_strided`。
* **示例**：`nn::conv2d_nhwc_int8_relu`

---

## 三、 数据库 Schema 设计 (SQLite)
系统以多张表存储核心数据，完全通过文本字段（结合 JSON 格式）实现对无限扩展参数的兼容。

### 1. 性能明细表 (`records`)
用于记录每一次 Benchmark 的原始测试数据，作为模型训练的样本库。

| 字段名 | 数据类型 | 描述说明 | 示例数据 |
| :--- | :--- | :--- | :--- |
| `op_name` | TEXT | 算子标准名称（见命名规范） | `"nn::conv2d_nchw_fp32"` |
| `device` | TEXT | 目标硬件标识 | `"NVIDIA_RTX_4090"` |
| `params` | TEXT (JSON) | **算法参数与内存状态的完整集合** | `{"N":1, "C":64, "is_contiguous":true, "memory_stride":[64,1]}` |
| `latency` | REAL | 实际执行的计算用时（目标预测值 $Y$） | `1.452` (ms) |
| `feature_order_key` | TEXT | 默认在录入时由 **`params` 自动推导**（与训练特征展开一致）；也可 API 手填或 `NULL`（未标注）。 |

### 2. 模型载体表 (`models`)
同一 `(op_name, device)` 下可并存**多个**模型，由 **`feature_order_key`** 区分（与训练得到的特征列序列一一对应；由 `feature_order` 列表经 `make_feature_order_key` 得到稳定字符串）。

| 字段名 | 数据类型 | 描述说明 |
| :--- | :--- | :--- |
| `op_name` | TEXT | 算子标准名称 (Primary Key Part 1) |
| `device` | TEXT | 目标硬件标识 (Primary Key Part 2) |
| `feature_order_key` | TEXT | 特征列顺序的规范串 (PK Part 3)，同序同键。 |
| `model_payload` | TEXT (JSON) | **XGBoost 森林结构的完整 JSON 序列化字符串**。 |
| `feature_order` | TEXT (JSON) | 训练时特征列名顺序，推理时与 `params` 对齐。 |

### 3. 参数模板表 (`param_templates`)
用于在 Web UI 或 API 中复用常用的 `params` JSON，与具体 `records` 无关。

| 字段名 | 数据类型 | 描述说明 |
| :--- | :--- | :--- |
| `id` | INTEGER | 自增主键 |
| `name` | TEXT | 模板名称（唯一） |
| `params` | TEXT (JSON) | 与 `records.params` 同结构的参数字典 |

---

## 四、 特征工程与数据降维机制 (Feature Engineering)
由于 XGBoost 只能接受一维数值向量 ($X$) 作为输入，数据库在提取 `records` 表的 `params` JSON 时，必须经过以下硬编码的自动化特征工程流水线：

### 1. 基础标量展平 (Flattening)
将 JSON 第一层的数值型参数（如 $N, C, H, W$, Kernel Size 等）直接解析为表格的独立特征列。

### 2. 内存步长展开 (Memory Stride Unfolding)
`memory_stride` 通常是一个列表（如 `[64, 1]`）。系统会根据列表长度，动态生成多列标量特征，例如：
* `memory_stride_0` = 64
* `memory_stride_1` = 1
缺失的维度使用 `0` 补齐，确保不同数据点的特征矩阵对齐。

### 3. 布尔值数值化 (Boolean Conversion)
将决定访存效率的核心物理状态 `is_contiguous` (True/False) 强制转化为整型特征 (1/0)。这对 XGBoost 极其重要，树模型会根据此特征直接进行节点分裂，惩罚不连续内存带来的额外开销。

### 4. 衍生特征 (可选但强烈建议)
在存入数据库前或特征提取时，计算并加入物理意义特征：
* **FLOPs (计算量)**：如矩阵乘的 $M \times N \times K$。
* **Memory_Bytes (访存量)**：输入输出张量的总体积。

---

## 五、 核心工作流与接口设计

系统通过 Streamlit 前端或底层 Python API 支持以下三种闭环工作流：

### 1. 自动录入与在线训练 (Auto-Fit)
1. 开发者输入算子名、设备、运行参数（包含 Shape 和 Stride 等）及实际耗时。
2. 数据写入 `records` 表。
3. 系统自动检索该 `(op_name, device)` 组合下的所有历史数据。
4. 如果数据量达到阈值（如 $\ge 2$），触发后台特征工程流水线。
5. 实例化 `xgb.XGBRegressor` 进行拟合。
6. 调用 `booster.save_raw('json')` 将模型转换为文本，以 `REPLACE INTO` 方式更新 `models` 表。

### 2. 离线接管与手动调优 (Manual Override)
针对需要超高精度的核心算子，系统提供干预接口：
1. **数据导出**：一键将某一算子的所有历史 JSON 记录导出为展平的 `.csv` 文件。
2. **离线炼丹**：开发者在本地 Jupyter 环境中使用该 CSV 进行深度网格搜索 (Grid Search)、清洗异常值 (Outliers)，训练出极致的 XGBoost 模型。
3. **参数回写**：将本地最优模型导出为 JSON 字符串，通过系统前端的“模型干预接口”直接粘贴并覆盖数据库中的 `model_payload`。

### 3. 编译器推理侧接口 (Inference API)
当外部调度器需要查询 Cost 时：
1. 传入待查询的 `op_name`, `device` 和当前参数字典。
2. 执行 `SELECT model_payload FROM models`。
3. 如果无模型，返回 `None`（触发调度器走真实 Benchmark 路线）。
4. 如果有模型，实例化空 XGBoost 引擎，使用 `load_model(bytearray(payload))` 反序列化。
5. 将传入的参数字典通过同样的特征工程流水线转换为向量，调用 `predict()` 返回预测耗时。

---

## 六、 系统的优势总结
1. **彻底解耦**：模型数据与关系数据同源同库，部署极其轻量，非常适合作为编译器的内嵌组件分发。
2. **拟合能力强**：XGBoost 能够完美拟合内存不连续 (`is_contiguous=0`) 或特定 Stride 下由于 Cache Miss 带来的非线性性能暴跌。
3. **灰度演进**：允许算子在没有模型时退化到启发式规则，随着运行数据累积自动演进为 AI 预测。

---

## 七、操作指导（本仓库实现）

以下说明对应仓库根目录下的 Python 包 `ocm/`、`app.py`（Streamlit）及默认 SQLite 路径。实现中在 `models` 表增加了 **`feature_order`** 字段（JSON 数组），用于保存训练时的特征列顺序，保证推理与训练对齐；其余设计与上文一致。

### 1. 环境与依赖

在已创建的 Conda 环境（例如名为 `ocm`）中，于**项目根目录** `ops_cost_model_database/` 执行：

```bash
conda activate ocm
pip install -r requirements.txt
```

依赖主要包括：`xgboost`、`scikit-learn`（`XGBRegressor` 需要）、`streamlit`、`pandas`、`numpy`。也可使用根目录的 `environment.yml` 通过 `conda env update -n ocm -f environment.yml` 同步（需在项目根目录执行）。

### 2. 启动 Web 管理界面（Streamlit）

```bash
cd /path/to/ops_cost_model_database
conda activate ocm
streamlit run app.py
```

浏览器打开终端提示的本地地址（一般为 `http://localhost:8501`）。侧栏可修改 **SQLite 数据库文件路径**；默认库文件为项目下 **`data/ocm.sqlite3`**（若不存在会在首次写入时自动创建目录与表结构）。**params 模板**存于当前库文件的 `param_templates` 表，换库即换模板列表。

界面分为多个标签页，建议按下列顺序使用：

| 标签页 | 用途 |
| :--- | :--- |
| **录入数据** | 填写 `op_name`、`device`、`params`（JSON）与 `latency`（ms）；**feature_order_key 由 params 自动生成**。高级选项可手填覆盖或写入未标注（NULL）。录入模式：仅写入，或写入后按**本条记录的 key** 筛选样本自动训练。 |
| **手动训练** | 选择 `(op_name, device)` 与**训练样本范围**：全部、仅未标注（`feature_order_key` 为空），或某一 key。 |
| **导出 CSV** | 选择 `(op_name, device)` 与 **feature_order_key 范围**（全部 / 仅未标注 / 某一 key）。CSV 含 `record_id`、`feature_order_key` 与展平后的参数列。 |
| **模型干预** | 粘贴离线得到的 `model_payload`（`booster.save_raw('json')` 的文本）与 **`feature_order`**（JSON 字符串数组），覆盖数据库中的模型。 |
| **推理试算** | 若同一 `(op_name, device)` 存在多个模型，需选择 **feature_order_key** 变体；仅一个模型时可自动选用。支持 params 模板加载与保存。 |

### 3. Python API 调用示例

- **仅录入**：直接调用 `insert_record(...)`，或 `add_record_maybe_autofit(..., auto_fit=False)`（默认即为不训练）。
- **录入并自动训练**：`add_record_maybe_autofit(..., auto_fit=True)`，或在多条 `insert_record` 之后调用 `fit_and_store_model(...)`。
- **params 模板**：`list_param_templates`、`get_param_template_by_name`、`save_param_template`、`delete_param_template`（名称唯一，同名保存则覆盖 `params`）。
- **多模型 / 特征模式**：`insert_record` 默认从 `params` 自动计算 `feature_order_key`（`derive_feature_order_key_from_params`），返回 `(row_id, key)`；`fit_and_store_model(..., feature_order_key=...)` 或 `unlabeled_only=True`；`predict_latency(..., feature_order_key=...)`；`list_models_for_op_device`、`get_model_row`、`make_feature_order_key`。

在**项目根目录**下将当前目录加入 `PYTHONPATH`，或使用 `pip install -e .`（若已配置可编辑安装）后导入 `ocm`：

```bash
cd /path/to/ops_cost_model_database
export PYTHONPATH="$PWD:$PYTHONPATH"
python -c "
from pathlib import Path
from ocm.database import get_connection, init_db, insert_record
from ocm.train import fit_and_store_model
from ocm.inference import predict_latency

conn = get_connection()  # 默认 data/ocm.sqlite3
init_db(conn)
_, fk = insert_record(conn, 'nn::matmul_row_major_fp32', 'GPU',
    {'M': 128, 'N': 256, 'K': 512, 'is_contiguous': True, 'memory_stride': [512, 1]}, 0.5)
insert_record(conn, 'nn::matmul_row_major_fp32', 'GPU',
    {'M': 256, 'N': 256, 'K': 512, 'is_contiguous': True, 'memory_stride': [512, 1]}, 0.9)
ok, msg = fit_and_store_model(conn, 'nn::matmul_row_major_fp32', 'GPU', feature_order_key=fk)
print(ok, msg)
print(predict_latency(conn, 'nn::matmul_row_major_fp32', 'GPU',
    {'M': 200, 'N': 200, 'K': 400, 'is_contiguous': True, 'memory_stride': [400, 1]},
    feature_order_key=fk))
conn.close()
"
```

推理时若 **`models` 中无对应模型**（含多模型时未指定 `feature_order_key` 且无法唯一确定），`predict_latency` 返回 `None`，编译器侧可回退到真实 Benchmark。

### 4. 测试目录说明

`test/OCMv1.py` 为可选入口脚本（将项目根加入 `sys.path` 后可配合 `streamlit run app.py` 使用）；架构说明与操作指引以仓库根目录 **`readme.md`**（本文）为准。
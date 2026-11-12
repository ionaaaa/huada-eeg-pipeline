# FBMAE / LaBraM / REVE 实验任务、入口命令与产物索引

本文档根据本目录中的腾讯云训练任务导出日志整理，日志下载清单为：

- `download_manifest_20261110_124317_093.json`
- 清单任务总数：38
- 成功并下载日志：36
- 失败并跳过：2（`train-1706671442327897856`、`train-1706627563731148672`）

## 重要说明：入口命令的保真边界

腾讯云导出的 `tar.gz` 只包含容器 stdout/stderr，不包含任务页面显示的 `Start command`。本批 36 份成功任务日志中没有任何一份保存 `Start command:`。

为满足“命令不要整理或修改”的要求，本文把日志中确实存在的 `resolved_command=` 值，以及后续从腾讯云任务页面补充的入口命令，逐字放入代码块，不根据环境变量摘要、脚本名或后续产物反推入口命令。仍未取得入口命令的任务明确标为“导出日志未保存”。因此：

- “FBMAE 原样命令”是从腾讯云任务页面另行取得并原样补充的入口命令；
- 后续模块中未标注 `resolved_command=` 的命令来自腾讯云任务页面；
- 标注“命令来源：日志 `resolved_command=`”的命令可由日志原文逐字核验；
- 仍标为“导出日志未保存”的任务，当前资料无法恢复其原始 shell 启动命令。

## 主流程总览

| 阶段 | FBMAE | LaBraM | REVE |
| --- | --- | --- | --- |
| Dortmund 源模型 | `train-1709457907739536128` | 源训练日志不在本批目录 | 源训练日志不在本批目录 |
| portable 导出 | 不适用本批 portable 流程 | `train-1713763624675313408` | `train-1713776953183569664` |
| dynamic parity | 不适用 | `train-1713824264320321280` | `train-1713834736683714304` |
| 华大原生 recording cache | 使用既有 FBMAE token Lance | `train-1713857675055466240` | `train-1713879267617670912` |
| strict zero-shot | `train-1709495944922652544` | `train-1713902939917125376` | `train-1713905320562132736` |
| probe CV7 | `train-1709512505019205376`（Dortmund probe transfer） | `train-1714404535918790400`（zero-init） | `train-1714405767367109376`（zero-init） |

## FBMAE 流程

### 1. 华大预处理后 token 与早期下游实验

| 任务 | 实验内容 | 主要输入/产物 | 入口命令 |
| --- | --- | --- | --- |
| `train-1706545103490909952` | 华大 8 被试 FBMAE token 提取，写入 480 行 | 输入：`/opt/ml/input/data/preprocessed_common/20261030_124625_manual_full/common_windows.lance`；产物：`/opt/ml/input/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance` | 见“FBMAE 原样命令 1” |
| `train-1706579874352062336` | 性别任务 manifest/split 与单次训练评估 | manifest：`/opt/ml/input/downstream_outputs/manifests/sex_20261030_151010/sex_subject_manifest.csv`；split：`/opt/ml/input/downstream_outputs/manifests/sex_20261030_151010/sex_subject_splits.csv`；checkpoint：`/opt/ml/input/downstream_outputs/eval_50m_sex_20261030_151834/checkpoints/downstream_finetune.pt` | 见“FBMAE 原样命令 2” |
| `train-1706603572785588992` | FBMAE 性别 CV4 | `/opt/ml/input/downstream_outputs/50m_sex_cv4_20261030_155717` | 见“FBMAE 原样命令 3” |
| `train-1708606274209449728` | cached-feature 性别 CV4 超参 sweep | `/opt/ml/input/downstream_outputs/50m_sex_cv4_sweep_20261102_101616`；feature cache：`/opt/ml/input/data/downstream_feature_cache/00709cbcb747eac398fff32b1e34134e178805b8167e0fb62bda61b64b6cef61` | 见“FBMAE 原样命令 4” |
| `train-1708649662956035968` | NeuralBench 性别 strict zero-shot；480 窗口、8 被试、60 次 forward | 日志未打印最终输出目录 | 见“FBMAE 原样命令 5” |
| `train-1708688254875862784` | NeuralBench 性别 source-probe transfer CV4 | `/opt/ml/input/downstream_outputs/50m_neuralbench_sex_probe_transfer_cv4_20261102_125953` | 见“FBMAE 原样命令 6” |
| `train-1708715114208823040` | 华大年龄旧版 cached-feature CV7 sweep | `/opt/ml/input/downstream_outputs/50m_age_cv7_sweep_20261102_135226` | 见“FBMAE 原样命令 7” |
| `train-1708745410111875840` | HBN 年龄 full-checkpoint zero-shot | `/opt/ml/input/downstream_outputs/50m_neuralbench_hbn_age_zero_shot_20261102_145241` | 见“FBMAE 原样命令 8” |
| `train-1708767420216379136` | 年龄收敛诊断，结果 `completed_not_converged` | `/opt/ml/input/downstream_outputs/50m_age_convergence_diagnostic_20261102_153623` | 见“FBMAE 原样命令 9” |
| `train-1708806799060240128` | subject-level Ridge nested LOSO | `/opt/ml/input/downstream_outputs/50m_age_subject_ridge_nested_loso_20261102_085444` | 见“FBMAE 原样命令 10” |

### 1.1 腾讯云任务页面补充的入口命令（原样）

以下命令按腾讯云任务页面提供的文本原样收录，未整理格式、补写注释、修改参数或改动路径。

#### FBMAE 原样命令 1：华大 8 被试 token 提取，把华大 EEG 转成 FBMAE 可用的 token 数据

- 任务：`train-1706545103490909952`

```bash
set -euo pipefail  PIPELINE_ROOT="/opt/ml/input" INPUT_LANCE="/opt/ml/input/data/preprocessed_common/20261030_124625_manual_full/common_windows.lance" RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)" OUTPUT_DIR="${PIPELINE_ROOT}/model_tokens_full/${RUN_TIMESTAMP}" OUTPUT_LANCE="${OUTPUT_DIR}/huada_model_tokens_8sub.lance"  test -f "${PIPELINE_ROOT}/scripts/build_huada_model_tokens.py" test -d "${INPUT_LANCE}"  python "${PIPELINE_ROOT}/scripts/build_huada_model_tokens.py" \   --input-lance-path "${INPUT_LANCE}" \   --output-lance-path "${OUTPUT_LANCE}"  echo "转换完成：${OUTPUT_DIR}"
```

#### FBMAE 原样命令 2：第一次华大性别单次划分实验，目的是验证“我们自己预训练 FBMAE 的ckpt能否用于华大性别分类”。
生成每个被试的 manifest（包括标签、split），训练性别线性头，独立评估，只有一次6/1/1 划分。
- 任务：`train-1706579874352062336`

```bash
set -euo pipefail
export HUADA_ROOT="/opt/ml/input" export PYTHON_BIN="python"
COS/CFS 已挂载到容器，本变量填写本地 Lance 目录。
export INPUT_LANCE="/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance"
export PRETRAINED_50M_CKPT="/opt/ml/input/data/checkpoint/pretrain_checkpoint_4.pt" export TIONE_OUTPUT_ROOT="/opt/ml/input/downstream_outputs"
cd "$HUADA_ROOT" test -f "$PRETRAINED_50M_CKPT" test -d "$INPUT_LANCE" || [[ "$INPUT_LANCE" == "://" ]] mkdir -p "$TIONE_OUTPUT_ROOT"
LANCE_AUTH=() if [[ -n "${STORAGE_OPTIONS:-}" ]]; then test -f "$STORAGE_OPTIONS" LANCE_AUTH=(--storage-options "$STORAGE_OPTIONS") fi
"$PYTHON_BIN" -c 'import torch,lance,pyarrow; print({"torch":torch.version,"cuda":torch.cuda.is_available(),"lance":lance.version,"pyarrow":pyarrow.version})'
generate manifest/split
STAMP="$(date +%Y%m%d_%H%M%S)" SEX_MANIFEST_DIR="$TIONE_OUTPUT_ROOT/manifests/sex_$STAMP"
"$PYTHON_BIN" scripts/create_downstream_manifest.py 
--input-lance "$INPUT_LANCE" 
"${LANCE_AUTH[@]}" 
--task sex_classification 
--seed 42 
--output-manifest "$SEX_MANIFEST_DIR/sex_subject_manifest.csv" 
--output-splits "$SEX_MANIFEST_DIR/sex_subject_splits.csv"
export SEX_SPLIT_MANIFEST="$SEX_MANIFEST_DIR/sex_subject_splits.csv"
#train : "${SEX_SPLIT_MANIFEST:?先生成并检查性别 split manifest}" STAMP="$(date +%Y%m%d_%H%M%S)" SEX_OUTPUT="$TIONE_OUTPUT_ROOT/50m_sex_full_$STAMP"
"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py 
--input-lance "$INPUT_LANCE" 
"${LANCE_AUTH[@]}" 
--task sex_classification 
--subject-split-manifest "$SEX_SPLIT_MANIFEST" 
--output-dir "$SEX_OUTPUT" 
--checkpoint "$PRETRAINED_50M_CKPT" 
--device cuda 
--mode linear_probe 
--epochs 50 
--batch-size 64 
--seed 42 
--split-seed 42
export SEX_DOWNSTREAM_CKPT="$SEX_OUTPUT/checkpoints/downstream_finetune.pt"
#eval : "${SEX_DOWNSTREAM_CKPT:?填写性别 downstream checkpoint}" STAMP="$(date +%Y%m%d_%H%M%S)"
"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py 
--input-lance "$INPUT_LANCE" 
"${LANCE_AUTH[@]}" 
--task sex_classification 
--subject-split-manifest "$SEX_SPLIT_MANIFEST" 
--output-dir "$TIONE_OUTPUT_ROOT/eval_50m_sex_$STAMP" 
--downstream-checkpoint "$SEX_DOWNSTREAM_CKPT" 
--device cuda 
--mode linear_probe 
--evaluate
```

#### FBMAE 原样命令 3：性别四折交叉验证，但入口命令后半段还重复跑了一次旧版单划分训练与评估。因此严格来说，它包含两个子实验。
- 任务：`train-1706603572785588992`

8 名被试必须恰好是 4 女、4 男。
先组成 4 个“1 女 + 1 男”的平衡被试对。
每一折使用：- 4 名被试训练，即 2 女 + 2 男
- 2 名被试验证，即 1 女 + 1 男
- 2 名被试测试，即 1 女 + 1 男


```bash
set -euo pipefail
export HUADA_ROOT="/opt/ml/input" export PYTHON_BIN="python"
COS/CFS 已挂载到容器，本变量填写本地 Lance 目录。
export INPUT_LANCE="/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance"
export PRETRAINED_50M_CKPT="/opt/ml/input/data/checkpoint/pretrain_checkpoint_4.pt" export TIONE_OUTPUT_ROOT="/opt/ml/input/downstream_outputs"
cd "$HUADA_ROOT" test -f "$PRETRAINED_50M_CKPT" test -d "$INPUT_LANCE" || [[ "$INPUT_LANCE" == "://" ]] mkdir -p "$TIONE_OUTPUT_ROOT"
LANCE_AUTH=() if [[ -n "${STORAGE_OPTIONS:-}" ]]; then test -f "$STORAGE_OPTIONS" LANCE_AUTH=(--storage-options "$STORAGE_OPTIONS") fi
"$PYTHON_BIN" -c 'import torch,lance,pyarrow; print({"torch":torch.version,"cuda":torch.cuda.is_available(),"lance":lance.version,"pyarrow":pyarrow.version})'
性别 4 折被试级交叉验证
STAMP="$(date +%Y%m%d_%H%M%S)" SEX_CV4_OUTPUT="$TIONE_OUTPUT_ROOT/50m_sex_cv4_$STAMP" test ! -e "$SEX_CV4_OUTPUT"
"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py 
--input-lance "$INPUT_LANCE" 
"${LANCE_AUTH[@]}" 
--task sex_classification 
--protocol sex_stratified_subject_cv4 
--folds 4 
--output-dir "$SEX_CV4_OUTPUT" 
--checkpoint "$PRETRAINED_50M_CKPT" 
--device cuda 
--mode linear_probe 
--epochs 5 
--batch-size 64 
--seed 42 
--split-seed 42
echo "CV4 output: $SEX_CV4_OUTPUT"
generate manifest/split
STAMP="$(date +%Y%m%d_%H%M%S)" SEX_MANIFEST_DIR="$TIONE_OUTPUT_ROOT/manifests/sex_$STAMP"
"$PYTHON_BIN" scripts/create_downstream_manifest.py 
--input-lance "$INPUT_LANCE" 
"${LANCE_AUTH[@]}" 
--task sex_classification 
--seed 42 
--output-manifest "$SEX_MANIFEST_DIR/sex_subject_manifest.csv" 
--output-splits "$SEX_MANIFEST_DIR/sex_subject_splits.csv"
export SEX_SPLIT_MANIFEST="$SEX_MANIFEST_DIR/sex_subject_splits.csv"
#train : "${SEX_SPLIT_MANIFEST:?先生成并检查性别 split manifest}" STAMP="$(date +%Y%m%d_%H%M%S)" SEX_OUTPUT="$TIONE_OUTPUT_ROOT/50m_sex_full_$STAMP"
"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py 
--input-lance "$INPUT_LANCE" 
"${LANCE_AUTH[@]}" 
--task sex_classification 
--protocol legacy_single_split 
--subject-split-manifest "$SEX_SPLIT_MANIFEST" 
--output-dir "$SEX_OUTPUT" 
--checkpoint "$PRETRAINED_50M_CKPT" 
--device cuda 
--mode linear_probe 
--epochs 5 
--batch-size 64 
--seed 42 
--split-seed 42
export SEX_DOWNSTREAM_CKPT="$SEX_OUTPUT/checkpoints/downstream_finetune.pt"
#eval : "${SEX_DOWNSTREAM_CKPT:?填写性别 downstream checkpoint}" STAMP="$(date +%Y%m%d_%H%M%S)"
"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py 
--input-lance "$INPUT_LANCE" 
"${LANCE_AUTH[@]}" 
--task sex_classification 
--protocol legacy_single_split 
--subject-split-manifest "$SEX_SPLIT_MANIFEST" 
--output-dir "$TIONE_OUTPUT_ROOT/eval_50m_sex_$STAMP" 
--downstream-checkpoint "$SEX_DOWNSTREAM_CKPT" 
--device cuda 
--mode linear_probe 
--evaluate
```

#### FBMAE 原样命令 4：高级版实验3，还是linear probe但是加了feature cache

- 任务：`train-1708606274209449728`

```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/input"
export PYTHON_BIN="python"
export PYTHONUNBUFFERED=1

export INPUT_LANCE="/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance"
export PRETRAINED_50M_CKPT="/opt/ml/input/data/checkpoint/pretrain_checkpoint_4.pt"
export SWEEP_CONFIG="/opt/ml/input/configs/50m_sex_cv4_sweep.json"
export TIONE_OUTPUT_ROOT="/opt/ml/input/downstream_outputs"

export FEATURE_CACHE_ROOT="/opt/ml/input/data/downstream_feature_cache"
export FEATURE_CACHE_POLICY="auto"
export FEATURE_CACHE_BATCH_SIZE="8"

export CACHE_LOG_EVERY_BATCHES="5"
export TRAIN_LOG_EVERY_EPOCHS="5"
export LOG_LEVEL="INFO"

export EPOCHS="100"
export BATCH_SIZE="64"
export TRAIN_SEED="42"
export SPLIT_SEED="42"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$PRETRAINED_50M_CKPT"
test -f "$SWEEP_CONFIG"
mkdir -p "$TIONE_OUTPUT_ROOT" "$FEATURE_CACHE_ROOT"

cd "$HUADA_ROOT"
bash commands/run_50m_sex_cv4_sweep_full.sh
```

#### FBMAE 原样命令 5：用NeuralBench 上sex数据集finetune过的ckpt对华大数据做 strict zero-shot

- 任务：`train-1708649662956035968`

```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/input"
export PYTHON_BIN="python"
export PYTHONUNBUFFERED=1

export INPUT_LANCE="/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance"
export NEURALBENCH_SEX_CKPT="/opt/ml/input/data/checkpoint/50M_sex_fft.ckpt"
export EXPECTED_CKPT_SHA256="56284992271d8cf0245bb6cc28f6098daf569d1188923196bfdd3d090a60342f"
export TIONE_OUTPUT_ROOT="/opt/ml/input/downstream_outputs"

export DEVICE="cuda"
export BATCH_SIZE="8"
export NUM_WORKERS="0"
export LOG_EVERY_BATCHES="5"
export LOG_LEVEL="INFO"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$NEURALBENCH_SEX_CKPT"
mkdir -p "$TIONE_OUTPUT_ROOT"

cd "$HUADA_ROOT"
bash commands/run_50m_neuralbench_sex_zero_shot_full.sh
```

#### FBMAE 原样命令 6：NeuralBench 外部性别 probe 在华大上的轻量迁移，从ft过的linear开始训练（不是从0开始），还是CV4

- 任务：`train-1708688254875862784`

```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/input"
export PYTHON_BIN="python"
export PYTHONUNBUFFERED=1

export INPUT_LANCE="/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance"
export NEURALBENCH_SEX_CKPT="/opt/ml/input/data/checkpoint/50M_sex_fft.ckpt"
export EXPECTED_CKPT_SHA256="56284992271d8cf0245bb6cc28f6098daf569d1188923196bfdd3d090a60342f"

export FEATURE_CACHE_ROOT="/opt/ml/input/downstream_feature_cache/50m_neuralbench_sex_mean"
export FEATURE_CACHE_POLICY="auto"
export FEATURE_CACHE_BATCH_SIZE="8"
export CACHE_LOG_EVERY_BATCHES="5"

export TIONE_OUTPUT_ROOT="/opt/ml/input/downstream_outputs"
export ZERO_SHOT_RESULTS_DIR="/opt/ml/input/downstream_outputs/50m_neuralbench_sex_zero_shot_20261102_114223"

export HEAD_LR="1e-5"
export WEIGHT_DECAY="0"
export EPOCHS="20"
export EARLY_STOPPING_PATIENCE="5"
export BATCH_SIZE="64"
export TRAIN_SEED="42"
export SPLIT_SEED="42"

export DEVICE="cuda"
export NUM_WORKERS="0"
export LOG_LEVEL="INFO"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$NEURALBENCH_SEX_CKPT"
test -d "$ZERO_SHOT_RESULTS_DIR"
mkdir -p "$FEATURE_CACHE_ROOT" "$TIONE_OUTPUT_ROOT"

cd "$HUADA_ROOT"
bash commands/run_50m_neuralbench_sex_probe_transfer_cv4_full.sh
```

#### FBMAE 原样命令 7：到了age任务，华大年龄 cached-feature CV7 超参数搜索

- 任务：`train-1708715114208823040`

```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/input"
export PYTHON_BIN="python"
export PYTHONUNBUFFERED=1

export INPUT_LANCE="/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance"
export PRETRAINED_50M_CKPT="/opt/ml/input/data/checkpoint/pretrain_checkpoint_4.pt"
export SWEEP_CONFIG="/opt/ml/input/configs/50m_age_cv7_sweep.json"
export FEATURE_CACHE_ROOT="/opt/ml/input/data/downstream_feature_cache"
export FEATURE_CACHE_BUILD_ROOT="/tmp/huada_downstream_feature_cache_build"
export FEATURE_CACHE_POLICY="auto"
export FEATURE_CACHE_BATCH_SIZE="8"
export TIONE_OUTPUT_ROOT="/opt/ml/input/downstream_outputs"

export DEVICE="cuda"
export EPOCHS="100"
export BATCH_SIZE="64"
export TRAIN_SEED="42"
export SPLIT_SEED="42"
export TRAIN_LOG_EVERY_EPOCHS="5"
export LOG_LEVEL="INFO"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$PRETRAINED_50M_CKPT"
test -f "$SWEEP_CONFIG"

mkdir -p "$TIONE_OUTPUT_ROOT" "$FEATURE_CACHE_ROOT" "$FEATURE_CACHE_BUILD_ROOT"

cd "$HUADA_ROOT"
bash commands/run_50m_age_cv7_sweep_full.sh
```

#### FBMAE 原样命令 8：用neural bench的age数据集（HBN）的ckpt在华大数据上做 zero-shot

- 任务：`train-1708745410111875840`

```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/input"
export PYTHON_BIN="python"
export PYTHONUNBUFFERED=1

export INPUT_LANCE="/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance"
export NEURALBENCH_HBN_AGE_CKPT="/opt/ml/input/data/checkpoint/50M_age_hbn_fft.ckpt"
export EXPECTED_CKPT_SHA256="a3438f5b13af4bb3a146e1c84f6ebdba22477d74046170825b98b2ef1be98b1d"
export TIONE_OUTPUT_ROOT="/opt/ml/input/downstream_outputs"

export DEVICE="cuda"
export BATCH_SIZE="8"
export NUM_WORKERS="0"
export LOG_EVERY_BATCHES="5"
export LOG_LEVEL="INFO"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$NEURALBENCH_HBN_AGE_CKPT"
mkdir -p "$TIONE_OUTPUT_ROOT"

cd "$HUADA_ROOT"
bash commands/run_50m_neuralbench_hbn_age_zero_shot_full.sh
```

#### FBMAE 原样命令 9：诊断任务 7 中年龄线性头的优化行为，包括小batch是否能过拟合以及全数据集是否收敛

- 任务：`train-1708767420216379136`

```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/input"
export PYTHON_BIN="python"
export PYTHONUNBUFFERED=1

export INPUT_LANCE="/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance"

export FEATURE_CACHE_DIR="/opt/ml/input/data/downstream_feature_cache/511f9d3fc52e2eaf5f18353df3ba39fa35a739602b6193e2b8a29beceaed8d35"
export MEAN_FEATURE_CACHE_LANCE="${FEATURE_CACHE_DIR}/mean_features.lance"
export FEATURE_CACHE_MANIFEST="${FEATURE_CACHE_DIR}/cache_manifest.json"

export TIONE_OUTPUT_ROOT="/opt/ml/input/downstream_outputs"

export FOLD="0"
export SPLIT_SEED="42"
export TRAIN_SEED="42"
export HEAD_LR="0.0003"
export WEIGHT_DECAY="0"
export BATCH_SIZE="64"
export MINI_SAMPLES_PER_SUBJECT="4"
export MINI_MAX_EPOCHS="1000"
export FULL_MAX_EPOCHS="300"
export STEP_LOG_INTERVAL="1"
export DEVICE="cuda"
export LOG_LEVEL="INFO"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -d "$MEAN_FEATURE_CACHE_LANCE"
test -f "$FEATURE_CACHE_MANIFEST"

mkdir -p "$TIONE_OUTPUT_ROOT"

cd "$HUADA_ROOT"
bash commands/run_50m_age_convergence_diagnostic_full.sh
```

#### FBMAE 原样命令 10：被试级 Ridge nested LOSO 年龄回归
- 先取每个窗口的 512 维 FBMAE mean feature。
- 对同一被试的全部窗口做算术平均。
- 最终得到 7 个独立样本，每个样本是一个 512 维被试级特征。
- SUB5 因年龄缺失排除。
模型不是神经网络训练头，而是 Ridge 回归：

- 任务：`train-1708806799060240128`

```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/input"
export PYTHON_BIN="python"
export PYTHONUNBUFFERED=1

export INPUT_LANCE="/opt/ml/input/data/model_tokens_full/20261030_140104/huada_model_tokens_8sub.lance"
export MEAN_FEATURE_CACHE_LANCE="/opt/ml/input/data/downstream_feature_cache/511f9d3fc52e2eaf5f18353df3ba39fa35a739602b6193e2b8a29beceaed8d35/mean_features.lance"
export FEATURE_CACHE_MANIFEST="/opt/ml/input/data/downstream_feature_cache/511f9d3fc52e2eaf5f18353df3ba39fa35a739602b6193e2b8a29beceaed8d35/cache_manifest.json"
export TIONE_OUTPUT_ROOT="/opt/ml/input/downstream_outputs"

export RIDGE_ALPHAS="0.01 0.1 1 10 100 1000"
export LOG_LEVEL="INFO"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -d "$MEAN_FEATURE_CACHE_LANCE"
test -f "$FEATURE_CACHE_MANIFEST"

mkdir -p "$TIONE_OUTPUT_ROOT"

cd "$HUADA_ROOT"
bash commands/run_50m_age_subject_ridge_nested_loso_full.sh
```

### 2. Dortmund 数据、token 与 full-finetune

| 任务 | 实验内容 | 主要输入/产物 | 入口命令 |
| --- | --- | --- | --- |
| `train-1709334220574426880` | Dortmund 年龄预处理 smoke | `/opt/ml/cfs_input/data/preprocess_common/dortmund_age_preprocessing_smoke_20261103_022233` | 见“Dortmund 原样命令 1” |
| `train-1709336860913960704` | Dortmund 年龄正式预处理 | `/opt/ml/cfs_input/data/preprocess_common/dortmund_age_preprocessing_full_20261103_022751` | 见“Dortmund 原样命令 2” |
| `train-1709366166407621376` | Dortmund FBMAE token smoke | `/opt/ml/cfs_input/data/model_tokens_full/fbmae_tokens_smoke/20261103_112600` | 见“Dortmund 原样命令 3” |
| `train-1709367807956879104` | Dortmund FBMAE token 正式生成 | `/opt/ml/cfs_input/data/model_tokens_full/fbmae_tokens/20261103_112918` | 见“Dortmund 原样命令 4” |
| `train-1709429609433148160` | Dortmund 年龄 full-finetune 两 epoch 单卡 smoke | split：`/opt/ml/cfs_input/downstream_output/dortmund_age_split_seed42_20261103_153000/subject_split_manifest.csv`；日志未打印 run output | 见“Dortmund 原样命令 5” |
| `train-1709430970937780992` | random-init 行为审计；日志明确 `random_init=true` | 同上 split；日志未打印 run output | 见“Dortmund 原样命令 6” |
| `train-1709436313533164288` | FBMAE 预训练 checkpoint 与 CUDA 条件审计，结果 `conditional_pass` | `/opt/ml/cos_input/checkpoint/pretrain_checkpoint_4.pt` | 见“Dortmund 原样命令 7” |
| `train-1709438326001501952` | Dortmund 年龄 full-finetune 双卡 DDP smoke | 日志仅保留两 epoch 指标，未打印 run output | 见“Dortmund 原样命令 8” |
| `train-1709445984546054016` | Dortmund 年龄 full-finetune 双卡 DDP smoke，显式校验预训练 checkpoint | checkpoint：`/opt/ml/cfs_input/data/checkpoint/pretrain_checkpoint_4.pt`；日志未打印 run output | 见“Dortmund 原样命令 9” |
| `train-1709457907739536128` | Dortmund 年龄正式 full-finetune；4 GPU、40 epoch 配置，运行到 early stopping | checkpoint 输入：`/opt/ml/cfs_input/data/checkpoint/pretrain_checkpoint_4.pt`；日志未打印最终 run output 路径 | 见“Dortmund 原样命令 10” |

### 2.1 腾讯云任务页面补充的入口命令（原样）

以下命令按腾讯云任务页面提供的文本原样收录，未整理格式、补写注释、修改参数或改动路径。

#### Dortmund 原样命令 1：年龄预处理 smoke

- 任务：`train-1709334220574426880`

```bash
set -euo pipefail

export HUADA\_ROOT="/opt/ml/cfs\_input"
export PYTHON\_BIN="python"
export DORTMUND\_DATA\_DIR="/opt/ml/cos\_input/data/Dortmund\_Vital\_Study"
export DORTMUND\_SOURCE\_URI="/opt/ml/cos\_input/data/Dortmund\_Vital\_Study"
export TIONE\_OUTPUT\_ROOT="/opt/ml/cfs\_input/data/preprocess\_common"

cd "$HUADA\_ROOT"
bash commands/run\_dortmund\_age\_preprocessing\_smoke.sh
```

#### Dortmund 原样命令 2：年龄正式预处理（没有做FBMAE适配）

先inspect得到sub和recording数量 

`train-1715953296965643008`
```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/cfs_input"
export PYTHON_BIN="python"
export DORTMUND_DATA_DIR="/opt/ml/cos_input/data/Dortmund_Vital_Study"
export DORTMUND_SOURCE_URI="/opt/ml/cos_input/data/Dortmund_Vital_Study"
export TIONE_OUTPUT_ROOT="/opt/ml/cfs_input/data/preprocess_common"


cd "$HUADA_ROOT"
python scripts/preprocess_dortmund_age_windows.py \
  --data-dir "$DORTMUND_DATA_DIR" \
  --config configs/huada_resting_preprocessing.toml \
  --output-dir "/opt/ml/cos_input/preprocessing_audit/dortmund_inventory_age20_44_$(date -u +%Y%m%d_%H%M%S)" \
  --min-age 20 \
  --max-age 44 \
  --cohort-id age20_44_ses1_pre_profile_a \
  --inventory-only
```


- 正式跑，任务：`train-1715957096208149248`

```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/cfs_input"
export PYTHON_BIN="python"
export DORTMUND_DATA_DIR="/opt/ml/cos_input/data/Dortmund_Vital_Study"
export DORTMUND_SOURCE_URI="/opt/ml/cos_input/data/Dortmund_Vital_Study"
export TIONE_OUTPUT_ROOT="/opt/ml/cfs_input/data/preprocess_common"
export DORTMUND_MIN_AGE=20
export DORTMUND_MAX_AGE=44
export DORTMUND_EXPECTED_SUBJECTS=291
export DORTMUND_EXPECTED_RECORDINGS=582
export DORTMUND_COHORT_ID="age20_44_ses1_pre_profile_a"

cd "$HUADA_ROOT"
bash commands/run_dortmund_age_preprocessing_full.sh
```

#### Dortmund 原样命令 3：把少量“公共预处理窗口”转换成 FBMAE 可以直接读取的 token 格式。


- 任务：`train-1709366166407621376`

```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/cfs_input"
export PYTHON_BIN="python"
export DORTMUND_COMMON_DIR="/opt/ml/cfs_input/data/preprocess_common/dortmund_age_preprocessing_age20_44_ses1_pre_profile_a_20261112_054059"
export DORTMUND_COMMON_LANCE="${DORTMUND_COMMON_DIR}/dataset.lance"
export DORTMUND_SOURCE_URI="cos://jianyu-1442740494/dortmund_age_preprocessing_age20_44_ses1_pre_profile_a_20261112_054059"
export ADAPTER_OUTPUT_DIR="/opt/ml/cfs_input/data/model_tokens_full/fbmae_tokens_smoke/$(date +%Y%m%d_%H%M%S)"
export ADAPTER_BATCH_SIZE="8"
export ADAPTER_NUM_WORKERS="4"

export DORTMUND_MIN_AGE=20
export DORTMUND_MAX_AGE=44
export DORTMUND_COHORT_ID="age20_44_ses1_pre_profile_a"
export DORTMUND_EXPECTED_SUBJECTS=291
export DORTMUND_EXPECTED_RECORDINGS=582

bash "${HUADA_ROOT}/commands/run_dortmund_fbmae_adapter_smoke.sh"
```

#### Dortmund 原样命令 4：FBMAE token 正式生成

- 任务：`train-1709367807956879104`

```bash
set -euo pipefail

export HUADA\_ROOT="/opt/ml/cfs\_input"
export PYTHON\_BIN="python"
export DORTMUND\_COMMON\_DIR="/opt/ml/cfs\_input/data/preprocess\_common/dortmund\_age\_preprocessing\_full\_20261103\_022751"
export DORTMUND\_COMMON\_LANCE=""
export DORTMUND\_SOURCE\_URI="cos\://jianyu-1442740494/huada/preprocessed\_common/dortmund\_age\_preprocessing\_full\_20261103\_022751/"
export ADAPTER\_OUTPUT\_DIR="/opt/ml/cfs\_input/data/model\_tokens\_full/fbmae\_tokens/$(date +%Y%m%d\_%H%M%S)"
export ADAPTER\_BATCH\_SIZE="32"
export ADAPTER\_NUM\_WORKERS="4"

bash "${HUADA\_ROOT}/commands/run\_dortmund\_fbmae\_adapter\_full.sh"
```

#### Dortmund 原样命令 5：随机初始化FBMAE full-finetune smoke，测试训练链路

- 任务：`train-1709429609433148160`

```bash
set -euo pipefail

export HUADA\_ROOT="/opt/ml/cfs\_input"
export PYTHON\_BIN="python"

cd "$HUADA\_ROOT"

export AUDIT\_DEPS="$HUADA\_ROOT/audit\_dortmund\_fbmae\_tokens/20261103\_124436/python\_deps"
export PYTHONPATH="$AUDIT\_DEPS:$HUADA\_ROOT/scripts/downstream\_50m:${PYTHONPATH:-}"

python - <<'PY'
import lance, pyarrow, torch
print("lance:", lance.**version**)
print("pyarrow:", pyarrow.**version**)
print("torch:", torch.**version**)
print("cuda devices:", torch.cuda.device\_count())
PY

export INPUT\_LANCE="$HUADA\_ROOT/data/model\_tokens\_full/fbmae\_tokens/20261103\_112918/dortmund\_fbmae\_tokens.lance"

export SPLIT\_ROOT="$HUADA\_ROOT/downstream\_output/dortmund\_age\_split\_seed42\_20261103\_153000"
export SPLIT\_MANIFEST="$SPLIT\_ROOT/subject\_split\_manifest.csv"

sha256sum "$SPLIT\_MANIFEST"

export AUDIT\_SUMMARY="$HUADA\_ROOT/audit\_dortmund\_fbmae\_tokens/20261103\_124436/audit\_summary.json"
export RUN\_OUTPUT="$HUADA\_ROOT/downstream\_output/dortmund\_age\_random\_init\_smoke\_$(date +%Y%m%d\_%H%M%S)"

python scripts/downstream\_50m/run\_dortmund\_age\_full\_finetune.py \
\--input-lance "$INPUT\_LANCE" \
\--split-manifest "$SPLIT\_MANIFEST" \
\--audit-summary "$AUDIT\_SUMMARY" \
\--output-dir "$RUN\_OUTPUT" \
\--epochs 2 \
\--batch-size 1 \
\--accumulation-steps 1 \
\--samples-per-epoch 2 \
\--backbone-lr 1e-5 \
\--head-lr 1e-4 \
\--weight-decay 0.05 \
\--gradient-clip-norm 1.0 \
\--scheduler linear\_warmup\_cosine\_decay \
\--warmup-ratio 0.1 \
\--early-stopping-patience 2 \
\--probe-layer-idx 8 \
\--pooling mean \
\--train-seed 42 \
\--amp-mode bf16 \
\--num-workers 0 \
\--smoke \
\--allow-random-init-smoke \
\--skip-test \
\--smoke-train-subjects 2 \
\--smoke-validation-subjects 2 \
\--smoke-windows-per-subject 1
```

#### Dortmund 原样命令 6：random-init 行为审计，单批次过拟合诊断，证明训练线路具备“学习能力”。

- 任务：`train-1709430970937780992`

```bash
set -euo pipefail

export HUADA\_ROOT="/opt/ml/cfs\_input"
export PYTHON\_BIN="python"

cd "$HUADA\_ROOT"

export AUDIT\_DEPS="$HUADA\_ROOT/audit\_dortmund\_fbmae\_tokens/20261103\_124436/python\_deps"
export PYTHONPATH="$AUDIT\_DEPS:$HUADA\_ROOT/scripts/downstream\_50m:${PYTHONPATH:-}"

python - <<'PY'
import lance, pyarrow, torch
print("lance:", lance.**version**)
print("pyarrow:", pyarrow.**version**)
print("torch:", torch.**version**)
print("cuda devices:", torch.cuda.device\_count())
PY

export INPUT\_LANCE="$HUADA\_ROOT/data/model\_tokens\_full/fbmae\_tokens/20261103\_112918/dortmund\_fbmae\_tokens.lance"

export SPLIT\_ROOT="$HUADA\_ROOT/downstream\_output/dortmund\_age\_split\_seed42\_20261103\_153000"
export SPLIT\_MANIFEST="$SPLIT\_ROOT/subject\_split\_manifest.csv"

sha256sum "$SPLIT\_MANIFEST"

export OVERFIT\_OUTPUT="$HUADA\_ROOT/downstream\_output/dortmund\_age\_overfit\_$(date +%Y%m%d\_%H%M%S)"

python scripts/downstream\_50m/run\_dortmund\_age\_single\_batch\_overfit.py \
\--input-lance "$INPUT\_LANCE" \
\--split-manifest "$SPLIT\_MANIFEST" \
\--output-dir "$OVERFIT\_OUTPUT" \
\--allow-random-init-smoke \
\--windows 16 \
\--micro-batch-size 2 \
\--steps 10 \
\--seed 42 \
\--amp-mode bf16

cat "$OVERFIT\_OUTPUT/overfit\_summary.json"
```

#### Dortmund 原样命令 7：预训练 checkpoint 与 CUDA 条件审计，不训练模型。

- 任务：`train-1709436313533164288`

```bash
set -euo pipefail

export HUADA\_ROOT="/opt/ml/cfs\_input"
export COS\_ROOT="/opt/ml/cos\_input"
export PYTHON\_BIN="python"

export INPUT\_LANCE="$HUADA\_ROOT/data/model\_tokens\_full/fbmae\_tokens/20261103\_112918/dortmund\_fbmae\_tokens.lance"
export SPLIT\_MANIFEST="$HUADA\_ROOT/downstream\_output/dortmund\_age\_split\_seed42\_20261103\_153000/subject\_split\_manifest.csv"

# 当前 conditional\_pass 审计，仅允许 smoke

export CONDITIONAL\_AUDIT\_SUMMARY="$HUADA\_ROOT/audit\_dortmund\_fbmae\_tokens/20261103\_124436/audit\_summary.json"

# 注意：这里必须是挂载后的普通文件路径

export PRETRAINED\_50M\_CKPT="$COS\_ROOT/checkpoint/pretrain\_checkpoint\_4.pt"

export OUTPUT\_ROOT="$HUADA\_ROOT/downstream\_output"
export EXPECTED\_CKPT\_SHA256="97335b696b3ae9138dcb51c736f49ee1c6008fdc22fc42f13ea9a5301452f36e"

cd "$HUADA\_ROOT"
export PYTHONPATH="$HUADA\_ROOT/scripts/downstream\_50m:${PYTHONPATH:-}"

hostname
pwd

test -d "$HUADA\_ROOT"
test -d "$INPUT\_LANCE"
test -f "$SPLIT\_MANIFEST"
test -f "$CONDITIONAL\_AUDIT\_SUMMARY"
test -f "$PRETRAINED\_50M\_CKPT"

ls -lh "$PRETRAINED\_50M\_CKPT"

printf '%s  %s\n' \
"$EXPECTED\_CKPT\_SHA256" \
"$PRETRAINED\_50M\_CKPT" \
\| sha256sum -c -

python - <<'PY'
import json
import os
import torch

audit = json.load(open(os.environ["CONDITIONAL\_AUDIT\_SUMMARY"]))
print("audit:", audit.get("overall\_conclusion"))
print("checkpoint:", os.environ["PRETRAINED\_50M\_CKPT"])
print("cuda\_count:", torch.cuda.device\_count())
for index in range(torch.cuda.device\_count()):
print(index, torch.cuda.get\_device\_name(index))
PY
```

#### Dortmund 原样命令 8：年龄 full-finetune 双卡 DDP smoke

- 任务：`train-1709438326001501952`

```bash
set -euo pipefail

export HUADA\_ROOT="/opt/ml/cfs\_input"
export COS\_ROOT="/opt/ml/cos\_input"
export PYTHON\_BIN="python"

export INPUT\_LANCE="$HUADA\_ROOT/data/model\_tokens\_full/fbmae\_tokens/20261103\_112918/dortmund\_fbmae\_tokens.lance"
export SPLIT\_MANIFEST="$HUADA\_ROOT/downstream\_output/dortmund\_age\_split\_seed42\_20261103\_153000/subject\_split\_manifest.csv"

# 当前 conditional\_pass 审计，仅允许 smoke

export CONDITIONAL\_AUDIT\_SUMMARY="$HUADA\_ROOT/audit\_dortmund\_fbmae\_tokens/20261103\_124436/audit\_summary.json"

# 注意：这里必须是挂载后的普通文件路径

export PRETRAINED\_50M\_CKPT="$COS\_ROOT/checkpoint/pretrain\_checkpoint\_4.pt"

export OUTPUT\_ROOT="$HUADA\_ROOT/downstream\_output"
export EXPECTED\_CKPT\_SHA256="97335b696b3ae9138dcb51c736f49ee1c6008fdc22fc42f13ea9a5301452f36e"

cd "$HUADA\_ROOT"
export PYTHONPATH="$HUADA\_ROOT/scripts/downstream\_50m:${PYTHONPATH:-}"



export NPROC\_PER\_NODE=2
export SMOKE\_OUTPUT="$OUTPUT\_ROOT/dortmund\_age\_pretrained\_ddp\_smoke\_$(date +%Y%m%d\_%H%M%S)"

test ! -e "$SMOKE\_OUTPUT"

torchrun \
\--standalone \
\--nproc\_per\_node="$NPROC\_PER\_NODE" \
scripts/downstream\_50m/run\_dortmund\_age\_full\_finetune.py \
\--input-lance "$INPUT\_LANCE" \
\--checkpoint "$PRETRAINED\_50M\_CKPT" \
\--split-manifest "$SPLIT\_MANIFEST" \
\--audit-summary "$CONDITIONAL\_AUDIT\_SUMMARY" \
\--output-dir "$SMOKE\_OUTPUT" \
\--epochs 2 \
\--batch-size 2 \
\--accumulation-steps 1 \
\--samples-per-epoch 8 \
\--backbone-lr 1e-5 \
\--head-lr 1e-4 \
\--weight-decay 0.05 \
\--gradient-clip-norm 1.0 \
\--scheduler linear\_warmup\_cosine\_decay \
\--warmup-ratio 0.1 \
\--early-stopping-patience 2 \
\--probe-layer-idx 8 \
\--pooling mean \
\--train-seed 42 \
\--amp-mode bf16 \
\--num-workers 0 \
\--smoke \
\--skip-test \
\--smoke-train-subjects 3 \
\--smoke-validation-subjects 2 \
\--smoke-windows-per-subject 4
```

#### Dortmund 原样命令 9：年龄 full-finetune 双卡 DDP smoke（显式 checkpoint）和8的区别是9调用.sh脚本，8是直接写出来了CLI

- 任务：`train-1709445984546054016`

```bash
set -euo pipefail

export HUADA\_ROOT="/opt/ml/cfs\_input"
export PRETRAINED\_50M\_CKPT="/opt/ml/cfs\_input/data/checkpoint/pretrain\_checkpoint\_4.pt"
export SPLIT\_MANIFEST="/opt/ml/cfs\_input/downstream\_output/dortmund\_age\_split\_seed42\_20261103\_153000/subject\_split\_manifest.csv"

export NPROC\_PER\_NODE="2"
export PER\_DEVICE\_BATCH\_SIZE="2"
export TRAIN\_SEED="42"

cd "$HUADA\_ROOT"
bash commands/run\_dortmund\_age\_full\_finetune\_smoke.sh
```

#### Dortmund 原样命令 10：年龄正式 full-finetune

- 任务：`train-1715993349867046656`

```bash
set -euo pipefail

export HUADA_ROOT="/opt/ml/cfs_input"
export PYTHON_BIN="python"

# 必须指向包含新增被试的新 token Lance
export INPUT_LANCE="$HUADA_ROOT/data/model_tokens_full/fbmae_tokens/20261112_142653/dortmund_fbmae_tokens.lance"

export PRETRAINED_50M_CKPT="$HUADA_ROOT/data/checkpoint/pretrain_checkpoint_4.pt"

export TRAIN_RATIO="0.8"
export VALIDATION_RATIO="0.1"
export TEST_RATIO="0.1"
export SPLIT_SEED="42"

# 必须生成新的冻结 split，不能继续使用旧的 214 人 manifest
export SPLIT_OUTPUT="$HUADA_ROOT/downstream_output/dortmund_age_split_seed${SPLIT_SEED}_$(date +%Y%m%d_%H%M%S)"

test -d "$HUADA_ROOT"
test -d "$INPUT_LANCE"
test -f "$PRETRAINED_50M_CKPT"
test ! -e "$SPLIT_OUTPUT"

cd "$HUADA_ROOT"

bash commands/create_dortmund_age_split.sh

export SPLIT_MANIFEST="$SPLIT_OUTPUT/subject_split_manifest.csv"
test -f "$SPLIT_MANIFEST"

# 强制确认 token Lance 和新 manifest 的被试集合完全一致
"$PYTHON_BIN" - "$INPUT_LANCE" "$SPLIT_MANIFEST" <<'PY'
import csv
import sys
import lance

lance_path, manifest_path = sys.argv[1:]

ds = lance.dataset(lance_path)
token_subjects = {
    str(x)
    for x in ds.to_table(columns=["subject_id"])["subject_id"].to_pylist()
}

with open(manifest_path, encoding="utf-8-sig", newline="") as f:
    manifest_rows = list(csv.DictReader(f))

manifest_subjects = {row["subject_id"].strip() for row in manifest_rows}

print("token_subject_count =", len(token_subjects))
print("manifest_subject_count =", len(manifest_subjects))
print("token_only =", sorted(token_subjects - manifest_subjects))
print("manifest_only =", sorted(manifest_subjects - token_subjects))

assert token_subjects == manifest_subjects, \
    "Token Lance and split manifest subject sets differ"

split_counts = {}
for row in manifest_rows:
    split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

print("split_counts =", split_counts)
assert all(split_counts.get(x, 0) > 0 for x in ("train", "validation", "test"))

print("NEW SPLIT MANIFEST VALIDATION PASSED")
PY

export NPROC_PER_NODE="4"
export PER_DEVICE_BATCH_SIZE="4"
export ACCUMULATION_STEPS="4"

export BACKBONE_LR="5e-6"
export HEAD_LR="5e-5"
export FULL_EPOCHS="40"

# Lance 必须保持 0
export NUM_WORKERS="0"

# 不要让新数据错误关联旧的审计报告。
# 如果已有针对新 token Lance 的审计，请把这里换成其 audit_summary.json。
# 这个不存在的路径会让当前 record_only 入口不附加旧审计。
export AUDIT_SUMMARY="$HUADA_ROOT/audit_dortmund_fbmae_tokens/NO_MATCHING_AUDIT_SUPPLIED.json"

bash commands/run_dortmund_age_full_finetune_full.sh
```

### 3. 华大 FBMAE strict zero-shot 与 probe transfer

| 任务 | 实验内容 | 产物 | 入口命令 |
| --- | --- | --- | --- |
| `train-1709495944922652544` | Dortmund full checkpoint 在华大 7 名有标签被试上的 strict zero-shot | 后续成功任务引用：`/opt/ml/cfs_input/downstream_output/50m_dortmund_age_huada_zero_shot_20261103_154351`；该任务自身日志只保留七阶段执行记录 | 见下方同任务入口命令 |
| `train-1709512505019205376` | Dortmund source probe 初始化的华大年龄 probe-transfer CV7 | `/opt/ml/cfs_input/downstream_output/50m_dortmund_age_probe_transfer_cv7_20261103_081646` | 见下方同任务入口命令 |

## LaBraM 流程

| 任务 | 实验内容 | 产物 | 入口命令 |
| --- | --- | --- | --- |
| `train-1713763624675313408` | Dortmund Age LaBraM FFT checkpoint 导出 portable v1 | `/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_labram/20261109T050258Z/dortmund_age_labram_fft_portable_v1.pt` | 见下方同任务入口命令 |
| `train-1713824264320321280` | 原始 FFT checkpoint 与 portable checkpoint 动态 parity | `/jianyu/neuroai-repro/repro_heavy_local/portable_parity/dortmund_age_labram/20261109T070329Z` | 见下方同任务入口命令 |
| `train-1713857675055466240` | 华大 BDF → LaBraM 原生 full-recording cache，8 被试 | `/mnt/cfs/jianyu/huada-eeg-pipeline/data/native_recording_cache/huada_labram/20261109T080946Z` | 见下方同任务入口命令 |
| `train-1713902939917125376` | Dortmund backbone + Dortmund age probe + source scaler 的华大 strict zero-shot | `/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_shot/huada_labram_dortmund_age_strict_zero_shot_20261109T093949Z` | 见下方同任务入口命令 |
| `train-1714404535918790400` | 相同 backbone/cache/CV7，age probe 零初始化训练 | `/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_init_probe_cv7/huada_labram_zero_init_probe_cv7_20261110T021622Z`；feature cache：`/mnt/cfs/jianyu/huada-eeg-pipeline/data/native_feature_cache/huada_labram_portable_age_mean` | 见下方同任务入口命令 |

## REVE 流程

| 任务 | 实验内容 | 产物 | 入口命令 |
| --- | --- | --- | --- |
| `train-1713776953183569664` | Dortmund Age REVE FFT checkpoint 导出 portable v1 | `/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_reve/20261109T052927Z/dortmund_age_reve_fft_portable_v1.pt` | 见下方同任务入口命令 |
| `train-1713834736683714304` | 原始 FFT checkpoint 与 portable checkpoint 动态 parity | `/jianyu/neuroai-repro/repro_heavy_local/portable_parity/dortmund_age_reve/20261109T072416Z` | 见下方同任务入口命令 |
| `train-1713879267617670912` | 华大 BDF → REVE 原生 full-recording cache，8 被试 | `/mnt/cfs/jianyu/huada-eeg-pipeline/data/native_recording_cache/huada_reve/20261109T085249Z` | 见下方同任务入口命令 |
| `train-1713905320562132736` | Dortmund backbone + Dortmund age probe + source scaler 的华大 strict zero-shot | `/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_shot/huada_reve_dortmund_age_strict_zero_shot_20261109T094439Z` | 见下方同任务入口命令 |
| `train-1714405767367109376` | 相同 backbone/cache/CV7，age probe 零初始化训练 | `/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_init_probe_cv7/huada_reve_zero_init_probe_cv7_20261110T021852Z`；feature cache：`/mnt/cfs/jianyu/huada-eeg-pipeline/data/native_feature_cache/huada_reve_portable_age_mean` | 见下方同任务入口命令 |

## 按流程模块排列的后续任务入口命令（原样）

以下入口按上方各模块的“任务｜实验内容｜产物”表格顺序排列。腾讯云任务页面提供的启动命令与日志中的 `resolved_command=` 合并展示；命令和已经确定的产物路径均保持原样。

### FBMAE：华大 strict zero-shot 与 probe transfer

#### `train-1709495944922652544`：Dortmund 年龄数据集在FBMAE上fft后在华大数据集上的 strict zero-shot

- 任务：`train-1709495944922652544`
- 产物：`/opt/ml/cfs_input/downstream_output/50m_dortmund_age_huada_zero_shot_20261103_154351`

```bash
set -euo pipefail

export HUADA\_ROOT="/opt/ml/cfs\_input"
export PYTHONPATH="$HUADA\_ROOT/scripts/downstream\_50m:${PYTHONPATH:-}"

cd "$HUADA\_ROOT"

RUN\_OUTPUT="$HUADA\_ROOT/downstream\_output/50m\_dortmund\_age\_huada\_zero\_shot\_$(date +%Y%m%d\_%H%M%S)"

python scripts/downstream\_50m/run\_huada\_dortmund\_age\_zero\_shot.py \
\--input-lance "/opt/ml/cfs\_input/model\_tokens\_full/20261030\_140104/huada\_model\_tokens\_8sub.lance" \
\--checkpoint "/opt/ml/cfs\_input/downstream\_output/dortmund\_age\_full\_finetune\_20261103\_142815/checkpoints/best\_full\_checkpoint.pt" \
\--expected-checkpoint-sha256 "3e44d9e7942d20d3d004e191a077ac636eeea770a6d185149591b683c5b3f13e" \
\--output-dir "$RUN\_OUTPUT" \
\--device cuda \
\--batch-size 8 \
\--num-workers 4 \
\--expected-source-age-mean 27.5 \
\--expected-source-age-std 4.218591
```

#### `train-1709512505019205376`：Dortmund source probe 初始化的华大年龄 probe-transfer CV7（不是从0训练probe）

- 任务：`train-1709512505019205376`
- 产物：`/opt/ml/cfs_input/downstream_output/50m_dortmund_age_probe_transfer_cv7_20261103_081646`

```bash
set -euo pipefail

export HUADA\_ROOT="/opt/ml/cfs\_input"
export PYTHON\_BIN="python"

export HUADA\_TOKEN\_LANCE="/opt/ml/cfs\_input/model\_tokens\_full/20261030\_140104/huada\_model\_tokens\_8sub.lance"
export DORTMUND\_AGE\_CKPT="/opt/ml/cfs\_input/downstream\_output/dortmund\_age\_full\_finetune\_20261103\_142815/checkpoints/best\_full\_checkpoint.pt"
export EXPECTED\_CKPT\_SHA256="3e44d9e7942d20d3d004e191a077ac636eeea770a6d185149591b683c5b3f13e"

export FEATURE\_CACHE\_ROOT="/opt/ml/cfs\_input/data/downstream\_feature\_cache"
export FEATURE\_CACHE\_BUILD\_ROOT="/tmp/huada\_dortmund\_age\_cache\_build"
export FEATURE\_CACHE\_POLICY="auto"
export FEATURE\_CACHE\_BATCH\_SIZE="8"

export OUTPUT\_ROOT="/opt/ml/cfs\_input/downstream\_output"

export DEVICE="cuda"
export NUM\_WORKERS="0"
export HEAD\_LR="1e-5"
export WEIGHT\_DECAY="0"
export EPOCHS="100"
export EARLY\_STOPPING\_PATIENCE="101"
export BATCH\_SIZE="64"
export TRAIN\_SEED="42"
export SPLIT\_SEED="42"

test -d "$HUADA\_ROOT"
test -d "$HUADA\_TOKEN\_LANCE"
test -f "$DORTMUND\_AGE\_CKPT"

mkdir -p "$FEATURE\_CACHE\_ROOT" "$OUTPUT\_ROOT" "$FEATURE\_CACHE\_BUILD\_ROOT"

cd "$HUADA\_ROOT"
bash commands/run\_huada\_dortmund\_age\_probe\_transfer\_cv7.sh
```

### LaBraM 流程

#### `train-1713763624675313408`：Dortmund 年龄 checkpoint 导出 portable ckpt
把 NeuralBench 中 Dortmund 年龄 full-finetune 的 LaBraM Lightning checkpoint，转换成项目约定的 portable_age_full_checkpoint v1

- 任务：`train-1713763624675313408`
- 产物：`/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_labram/20261109T050258Z/dortmund_age_labram_fft_portable_v1.pt`

```bash
bash -lc 'set -Eeuo pipefail
exec 2>&1

REPO=/jianyu/neuroai-repro
SOURCE\_RESULT\_DIR="$REPO/repro\_heavy\_local/save/dortmund\_age\_labram\_fft\_full/20261105T070714Z"
SOURCE\_CHECKPOINT="$SOURCE\_RESULT\_DIR/labram/best\_full\_checkpoint.ckpt"

cd "$REPO"

SOURCE\_RESULT\_DIR="$SOURCE\_RESULT\_DIR" \
SOURCE\_CHECKPOINT="$SOURCE\_CHECKPOINT" \
PORTABLE\_ROOT="$REPO/repro\_heavy\_local/portable\_checkpoints" \
EXPORT\_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)" \
bash neuralbench-repo/commands/export\_dortmund\_age\_labram\_portable.sh
'
```

#### `train-1713824264320321280`：原始 FFT checkpoint 与 portable checkpoint 动态 parity
验证 portable 导出有没有改变模型行为。同时加载原始ckpt和刚导出的portable ckpt，在相同input上运行两个，比较输出

- 任务：`train-1713824264320321280`
- 产物：`/jianyu/neuroai-repro/repro_heavy_local/portable_parity/dortmund_age_labram/20261109T070329Z`
- 命令来源：日志 `resolved_command=`

```bash
/opt/venv/bin/python /jianyu/neuroai-repro/neuralbench-repo/neuralbench/checkpoints/portable_age_dynamic_parity.py --model-family labram --source-checkpoint /jianyu/neuroai-repro/repro_heavy_local/save/dortmund_age_labram_fft_full/20261105T070714Z/labram/best_full_checkpoint.ckpt --expected-source-checkpoint-sha256 679a113ec8abf105b87e8acc95a9648fd946976e515c8a9f3aa212377b6dfbb0 --portable-checkpoint /jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_labram/20261109T050258Z/dortmund_age_labram_fft_portable_v1.pt --expected-portable-checkpoint-sha256 7cc7b1680a4fb9f82b970c98bd6d7688a4813ca9752e57d0eb4d9e94d2d7e998 --portable-sidecar /jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_labram/20261109T050258Z/dortmund_age_labram_fft_portable_v1.sidecar.json --input-contract /jianyu/neuroai-repro/repro_heavy_local/save/dortmund_age_labram_fft_full/20261105T070714Z/labram/input_contract_snapshot.json --expected-input-contract-sha256 d6c6aa3c745dfaac83c0b931c6327f16b493672ebe52902c96e16fac24cca579 --resolved-config /jianyu/neuroai-repro/repro_heavy_local/save/dortmund_age_labram_fft_full/20261105T070714Z/labram/resolved_config.yaml --pretrained-snapshot /opt/ml/neurobench/cache/huggingface/hub/models--braindecode--labram-pretrained/snapshots/0563b6c626e7b40d9a36653b763715db94d945d7 --output-dir /jianyu/neuroai-repro/repro_heavy_local/portable_parity/dortmund_age_labram/20261109T070329Z --device cuda:0 --batch-size 2 --seed 20261109 --atol 1e-5 --rtol 1e-5
```

#### `train-1713857675055466240`：华大 BDF 生成原生 full-recording cache
把 8 名华大受试者的原始 BDF 转换成 LaBraM 自己需要的输入格式。

- 任务：`train-1713857675055466240`
- 产物：`/mnt/cfs/jianyu/huada-eeg-pipeline/data/native_recording_cache/huada_labram/20261109T080946Z`

````bash
bash -lc '
set -Eeuo pipefail
exec 2>&1

export HUADA\_ROOT=/mnt/cfs/jianyu/huada-eeg-pipeline
export PYTHON\_BIN=/opt/venv/bin/python
export PYTHONPATH="$HUADA\_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

export MODEL\_FAMILY=labram
export TARGET\_CONTRACT="$HUADA\_ROOT/configs/huada\_labram\_target\_input\_contract\_v1.json"
export EXPECTED\_TARGET\_CONTRACT\_FILE\_SHA256=fd943b1a620cbb66cfedb40a18b065eed97cbc4be256fe86758b0cad32634e57

export RUN\_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export CACHE\_ROOT="$HUADA\_ROOT/data/native\_recording\_cache/huada\_labram"
export CACHE\_RUN\_ROOT="$CACHE\_ROOT/$RUN\_ID"
export LOG\_ROOT="$CACHE\_ROOT/bootstrap\_logs"
export LOG\_FILE="$LOG\_ROOT/${RUN\_ID}\_labram\_native\_cache.log"

test -x "$PYTHON\_BIN"
test -d "$HUADA\_ROOT/scripts/native\_backends"
test -f "$TARGET\_CONTRACT"

OBSERVED\_CONTRACT\_SHA256="$(sha256sum "$TARGET\_CONTRACT" | awk "{print $1}")"
[[ "$OBSERVED\_CONTRACT\_SHA256" == "$EXPECTED\_TARGET\_CONTRACT\_FILE\_SHA256" ]] || {
echo "ERROR: LaBraM target contract SHA mismatch"
echo "expected=$EXPECTED\_TARGET\_CONTRACT\_FILE\_SHA256"
echo "actual=$OBSERVED\_CONTRACT\_SHA256"
exit 2
}

HUADA\_BDF\_ROOT="/opt/ml/huada/data/resting\_state"

if [[ -z "$HUADA\_BDF\_ROOT" ]]; then
for candidate in \
/opt/ml/huada/data \
/opt/ml/input/data \
/opt/ml/input\_cfs \
"$HUADA\_ROOT/data"
do
echo "checking\_bdf\_root=$candidate"
if [[ -d "$candidate/SUB1\_31\_M" ]]; then
HUADA\_BDF\_ROOT="$candidate"
break
fi
done
fi

[[ -n "$HUADA\_BDF\_ROOT" && -d "$HUADA\_BDF\_ROOT" ]] || {
echo "ERROR: Huada BDF root not found"
echo "Set HUADA\_BDF\_ROOT to the directory containing SUB1\_31\_M ... SUB8\_22\_M"
exit 2
}

SUBJECT\_IDS=(SUB1 SUB2 SUB3 SUB4 SUB5 SUB6 SUB7 SUB8)
SUBJECT\_DIRS=(
SUB1\_31\_M
SUB2\_29\_F
SUB3\_26\_F
SUB4\_26\_F
SUB5\_M
SUB6\_28\_M
SUB7\_28\_F
SUB8\_22\_M
)

[[ ! -e "$CACHE\_RUN\_ROOT" ]] || {
echo "ERROR: cache run already exists: $CACHE\_RUN\_ROOT"
exit 2
}

mkdir -p "$CACHE\_RUN\_ROOT/recordings" "$LOG\_ROOT"

echo "huada\_root=$HUADA\_ROOT"
echo "huada\_bdf\_root=$HUADA\_BDF\_ROOT"
echo "target\_contract=$TARGET\_CONTRACT"
echo "target\_contract\_sha256=$OBSERVED\_CONTRACT\_SHA256"
echo "cache\_run\_root=$CACHE\_RUN\_ROOT"
echo "log\_file=$LOG\_FILE"

set +e
(
set -Eeuo pipefail
cd "$HUADA\_ROOT"
```bash
"$PYTHON_BIN" -c "
```

import mne
import numpy
import scipy
print("python dependencies: OK")
print("mne=", mne.**version**)
print("numpy=", numpy.**version**)
print("scipy=", scipy.**version**)
"
```typescript
for index in "${!SUBJECT_IDS[@]}"; do
    recording_id="${SUBJECT_IDS[$index]}"
    subject_dir="$HUADA_BDF_ROOT/${SUBJECT_DIRS[$index]}"

    mapfile -t bdf_files < <(
        find "$subject_dir" -type f -iname "*.bdf" -print | sort
    )

    [[ "${#bdf_files[@]}" -eq 1 ]] || {
        echo "ERROR: expected exactly one BDF under $subject_dir"
        printf "candidate=%s\n" "${bdf_files[@]:-NONE}"
        exit 2
    }

    source_bdf="${bdf_files[0]}"
    output_dir="$CACHE_RUN_ROOT/recordings/$recording_id"

    echo "building recording_id=$recording_id"
    echo "source_bdf=$source_bdf"
    echo "output_dir=$output_dir"

    "$PYTHON_BIN" -m scripts.native_backends.build_huada_native_recording_cache \
        --model-family labram \
        --source-bdf "$source_bdf" \
        --recording-id "$recording_id" \
        --target-input-contract "$TARGET_CONTRACT" \
        --expected-target-contract-file-sha256 "$EXPECTED_TARGET_CONTRACT_FILE_SHA256" \
        --output-dir "$output_dir"

    test -f "$output_dir/_SUCCESS"
done

printf "%s\n" \
    "model_family=labram" \
    "run_id=$RUN_ID" \
    "recording_count=${#SUBJECT_IDS[@]}" \
    "target_contract_sha256=$OBSERVED_CONTRACT_SHA256" \
    > "$CACHE_RUN_ROOT/_SUCCESS"

echo "LaBraM native cache completed: $CACHE_RUN_ROOT"
```

) 2>&1 | tee "$LOG\_FILE"

status=${PIPESTATUS[0]}
set -e

echo "exit\_code=$status"
echo "log\_file=$LOG\_FILE"
echo "cache\_run\_root=$CACHE\_RUN\_ROOT"
exit "$status"
'
````

#### `train-1713902939917125376`：Dortmund backbone + age probe + source scaler 的华大 strict zero-shot

- 任务：`train-1713902939917125376`
- 产物：`/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_shot/huada_labram_dortmund_age_strict_zero_shot_20261109T093949Z`
- 命令来源：日志 `resolved_command=`

```bash
/opt/venv/bin/python -m scripts.native_backends.run_huada_portable_age_zero_shot --model-family labram --native-cache-run-root /mnt/cfs/jianyu/huada-eeg-pipeline/data/native_recording_cache/huada_labram/20261109T080946Z --subject-manifest /mnt/cfs/jianyu/huada-eeg-pipeline/preprocessing_audit/20261030_105520/subject_manifest.json --target-input-contract /mnt/cfs/jianyu/huada-eeg-pipeline/configs/huada_labram_target_input_contract_v1.json --expected-target-contract-file-sha256 fd943b1a620cbb66cfedb40a18b065eed97cbc4be256fe86758b0cad32634e57 --portable-checkpoint /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_labram/20261109T050258Z/dortmund_age_labram_fft_portable_v1.pt --portable-sidecar /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_labram/20261109T050258Z/dortmund_age_labram_fft_portable_v1.sidecar.json --expected-portable-checkpoint-sha256 7cc7b1680a4fb9f82b970c98bd6d7688a4813ca9752e57d0eb4d9e94d2d7e998 --expected-source-checkpoint-sha256 679a113ec8abf105b87e8acc95a9648fd946976e515c8a9f3aa212377b6dfbb0 --dynamic-parity-dir /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_parity/dortmund_age_labram/20261109T070329Z --output-dir /mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_shot/huada_labram_dortmund_age_strict_zero_shot_20261109T093949Z --device cuda:0 --batch-size 8 --expected-source-age-mean 27.5 --expected-source-age-std 4.218591213226318 --comparison-result fbmae=/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/50m_dortmund_age_huada_zero_shot_20261103_154351
```

#### `train-1714404535918790400`：相同 backbone/cache/CV7 的 age probe 零初始化训练

- 任务：`train-1714404535918790400`
- 产物：`/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_init_probe_cv7/huada_labram_zero_init_probe_cv7_20261110T021622Z`
- feature cache：`/mnt/cfs/jianyu/huada-eeg-pipeline/data/native_feature_cache/huada_labram_portable_age_mean`
- 命令来源：日志 `resolved_command=`

```bash
/opt/venv/bin/python -m scripts.native_backends.run_huada_portable_age_zero_init_probe_cv7 --model-family labram --native-cache-run-root /mnt/cfs/jianyu/huada-eeg-pipeline/data/native_recording_cache/huada_labram/20261109T080946Z --subject-manifest /mnt/cfs/jianyu/huada-eeg-pipeline/preprocessing_audit/20261030_105520/subject_manifest.json --target-input-contract /mnt/cfs/jianyu/huada-eeg-pipeline/configs/huada_labram_target_input_contract_v1.json --expected-target-contract-file-sha256 fd943b1a620cbb66cfedb40a18b065eed97cbc4be256fe86758b0cad32634e57 --portable-checkpoint /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_labram/20261109T050258Z/dortmund_age_labram_fft_portable_v1.pt --portable-sidecar /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_labram/20261109T050258Z/dortmund_age_labram_fft_portable_v1.sidecar.json --expected-portable-checkpoint-sha256 7cc7b1680a4fb9f82b970c98bd6d7688a4813ca9752e57d0eb4d9e94d2d7e998 --expected-source-checkpoint-sha256 679a113ec8abf105b87e8acc95a9648fd946976e515c8a9f3aa212377b6dfbb0 --expected-source-age-mean 27.5 --expected-source-age-std 4.218591213226318 --dynamic-parity-dir /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_parity/dortmund_age_labram/20261109T070329Z --strict-zero-shot-reference-dir /mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_shot/huada_labram_dortmund_age_strict_zero_shot_20261109T093949Z --feature-cache-root /mnt/cfs/jianyu/huada-eeg-pipeline/data/native_feature_cache/huada_labram_portable_age_mean --feature-cache-policy auto --output-dir /mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_init_probe_cv7/huada_labram_zero_init_probe_cv7_20261110T021622Z --device cuda:0 --feature-cache-batch-size 8 --batch-size 64 --head-lr 1e-5 --weight-decay 0 --epochs 100 --early-stopping-patience 15 --seed 42 --split-seed 42
```

### REVE 流程

#### `train-1713776953183569664`：Dortmund 年龄 checkpoint 导出 portable v1

- 任务：`train-1713776953183569664`
- 产物：`/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_reve/20261109T052927Z/dortmund_age_reve_fft_portable_v1.pt`

```bash
bash -lc 'set -Eeuo pipefail
exec 2>&1

REPO=/jianyu/neuroai-repro
SOURCE\_RESULT\_DIR="$REPO/repro\_heavy\_local/save/dortmund\_age\_reve\_fft\_full/20261106T100131Z"
SOURCE\_CHECKPOINT="$SOURCE\_RESULT\_DIR/reve/best\_full\_checkpoint.ckpt"
INPUT\_CONTRACT="$SOURCE\_RESULT\_DIR/reve/input\_contract\_snapshot.json"
SPLIT\_MANIFEST="$SOURCE\_RESULT\_DIR/reve/subject\_split\_manifest.csv"

test -f "$SOURCE\_CHECKPOINT"
test ! -L "$SOURCE\_CHECKPOINT"
test "$(stat -c %s "$SOURCE\_CHECKPOINT")" = "831621213"
test -f "$INPUT\_CONTRACT"
test -f "$SPLIT\_MANIFEST"

cd "$REPO"

SOURCE\_RESULT\_DIR="$SOURCE\_RESULT\_DIR" \
SOURCE\_CHECKPOINT="$SOURCE\_CHECKPOINT" \
INPUT\_CONTRACT="$INPUT\_CONTRACT" \
SPLIT\_MANIFEST="$SPLIT\_MANIFEST" \
PORTABLE\_ROOT="$REPO/repro\_heavy\_local/portable\_checkpoints" \
EXPORT\_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)" \
bash neuralbench-repo/commands/export\_dortmund\_age\_reve\_portable.sh
'
```

#### `train-1713834736683714304`：原始 FFT checkpoint 与 portable checkpoint 动态 parity

- 任务：`train-1713834736683714304`
- 产物：`/jianyu/neuroai-repro/repro_heavy_local/portable_parity/dortmund_age_reve/20261109T072416Z`
- 命令来源：日志 `resolved_command=`

```bash
/opt/venv/bin/python /jianyu/neuroai-repro/neuralbench-repo/neuralbench/checkpoints/portable_age_dynamic_parity.py --model-family reve --source-checkpoint /jianyu/neuroai-repro/repro_heavy_local/save/dortmund_age_reve_fft_full/20261106T100131Z/reve/best_full_checkpoint.ckpt --expected-source-checkpoint-sha256 ca3474382da2f03937232eedcdbaf7faf511725af2542561eeb5e7e0d5631201 --portable-checkpoint /jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_reve/20261109T052927Z/dortmund_age_reve_fft_portable_v1.pt --expected-portable-checkpoint-sha256 e2a7f72d435e7c8bc7688e7c27d0db5cccbc226b8dcf0acaf21172a38ed8fff4 --portable-sidecar /jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_reve/20261109T052927Z/dortmund_age_reve_fft_portable_v1.sidecar.json --input-contract /jianyu/neuroai-repro/repro_heavy_local/save/dortmund_age_reve_fft_full/20261106T100131Z/reve/input_contract_snapshot.json --expected-input-contract-sha256 f72f1e8ff93a9f51ed9d76a499019dd092a892bd308bbfc8e974d406b300c092 --resolved-config /jianyu/neuroai-repro/repro_heavy_local/save/dortmund_age_reve_fft_full/20261106T100131Z/reve/resolved_config.yaml --pretrained-snapshot /opt/ml/neurobench/cache/huggingface/hub/models--brain-bzh--reve-base/snapshots/fa9a2163a4b7c0a42c8e28b56077ef9c368944dc --output-dir /jianyu/neuroai-repro/repro_heavy_local/portable_parity/dortmund_age_reve/20261109T072416Z --device cuda:0 --batch-size 2 --seed 20261109 --atol 1e-5 --rtol 1e-5 --reve-positions-file /root/mne_data/reve_positions.json
```

#### `train-1713879267617670912`：华大 BDF 生成原生 full-recording cache

- 任务：`train-1713879267617670912`
- 产物：`/mnt/cfs/jianyu/huada-eeg-pipeline/data/native_recording_cache/huada_reve/20261109T085249Z`

````bash
bash -lc '
set -Eeuo pipefail
exec 2>&1

export HUADA\_ROOT=/mnt/cfs/jianyu/huada-eeg-pipeline
export PYTHON\_BIN=/opt/venv/bin/python
export PYTHONPATH="$HUADA\_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

export MODEL\_FAMILY=reve
export HUADA\_BDF\_ROOT=/opt/ml/huada/data/resting\_state

export REVE\_POSITIONS\_SOURCE=/opt/ml/input/data/checkpoints/omni/reve-positions/positions.json
export EXPECTED\_REVE\_POSITIONS\_SHA256=f62ca8d7dd4b88e1422ea17348440a41ed0b73b9ff7d12c3b1229492c9b448a5

export RUN\_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export CACHE\_ROOT="$HUADA\_ROOT/data/native\_recording\_cache/huada\_reve"
export CACHE\_RUN\_ROOT="$CACHE\_ROOT/$RUN\_ID"
export RECORDING\_CACHE\_ROOT="$CACHE\_RUN\_ROOT/recordings"
export TARGET\_CONTRACT="$CACHE\_RUN\_ROOT/huada\_reve\_target\_input\_contract\_v1.json"
export CHANNEL\_NAME\_MAPPING="$CACHE\_RUN\_ROOT/reve\_channel\_name\_mapping.json"
export LOG\_ROOT="$CACHE\_ROOT/bootstrap\_logs"
export LOG\_FILE="$LOG\_ROOT/${RUN\_ID}\_reve\_native\_cache.log"

test -x "$PYTHON\_BIN"
test -d "$HUADA\_ROOT"
test -d "$HUADA\_ROOT/scripts/native\_backends"
test -d "$HUADA\_BDF\_ROOT"
test -s "$REVE\_POSITIONS\_SOURCE"
test ! -L "$REVE\_POSITIONS\_SOURCE"

position\_sha\_line="$(sha256sum "$REVE\_POSITIONS\_SOURCE")"
observed\_position\_sha="${position\_sha\_line%% \*}"

echo "reve\_positions\_source=$REVE\_POSITIONS\_SOURCE"
echo "expected\_position\_sha256=$EXPECTED\_REVE\_POSITIONS\_SHA256"
echo "observed\_position\_sha256=$observed\_position\_sha"

[[ "$observed\_position\_sha" == "$EXPECTED\_REVE\_POSITIONS\_SHA256" ]] || {
echo "ERROR: REVE position asset SHA mismatch"
exit 2
}

"$PYTHON\_BIN" -c "
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
positions = json.load(handle)

for name in ("PO9", "PO10"):
if name not in positions:
raise SystemExit(f"required mapped position is absent: {name}")
value = positions[name]
if not isinstance(value, list) or len(value) != 3:
raise SystemExit(f"invalid position value for {name}: {value!r}")

print("mapped position names verified: PO9 PO10")
print("PO9=", positions["PO9"])
print("PO10=", positions["PO10"])
" "$REVE\_POSITIONS\_SOURCE"

SUBJECT\_IDS=(
SUB1
SUB2
SUB3
SUB4
SUB5
SUB6
SUB7
SUB8
)

SOURCE\_BDFS=(
"$HUADA\_BDF\_ROOT/SUB1\_31\_M/BGI\_EEG\_20261023172343/20261023172343.bdf"
"$HUADA\_BDF\_ROOT/SUB2\_29\_F/BGI\_EEG\_20261026150143/20261026150143.bdf"
"$HUADA\_BDF\_ROOT/SUB3\_26\_F/BGI\_EEG\_20261026152405/20261026152405.bdf"
"$HUADA\_BDF\_ROOT/SUB4\_26\_F/BGI\_EEG\_20261026155541/20261026155541.bdf"
"$HUADA\_BDF\_ROOT/SUB5\_M/BGI\_EEG\_20261026161903/20261026161903.bdf"
"$HUADA\_BDF\_ROOT/SUB6\_28\_M/BGI\_EEG\_20261026163827/20261026163827.bdf"
"$HUADA\_BDF\_ROOT/SUB7\_28\_F/BGI\_EEG\_20261029145112/20261029145112.bdf"
"$HUADA\_BDF\_ROOT/SUB8\_22\_M/BGI\_EEG\_20261029160859/20261029160859.bdf"
)

[[ "${#SUBJECT\_IDS[@]}" -eq "${#SOURCE\_BDFS[@]}" ]] || {
echo "ERROR: subject/BDF array length mismatch"
exit 2
}

for index in "${!SUBJECT\_IDS[@]}"; do
subject\_id="${SUBJECT\_IDS[$index]}"
source\_bdf="${SOURCE\_BDFS[$index]}"
```bash
[[ -f "$source_bdf" && ! -L "$source_bdf" ]] || {
    echo "ERROR: missing or unsafe BDF"
    echo "subject_id=$subject_id"
    echo "source_bdf=$source_bdf"
    exit 2
}
```

done

[[ ! -e "$CACHE\_RUN\_ROOT" ]] || {
echo "ERROR: output already exists: $CACHE\_RUN\_ROOT"
exit 2
}

mkdir -p "$RECORDING\_CACHE\_ROOT" "$LOG\_ROOT"

printf "%s\n" \
"{"CB1":"PO9","CB2":"PO10"}" \
\> "$CHANNEL\_NAME\_MAPPING"

test -s "$CHANNEL\_NAME\_MAPPING"

echo "huada\_root=$HUADA\_ROOT"
echo "huada\_bdf\_root=$HUADA\_BDF\_ROOT"
echo "run\_id=$RUN\_ID"
echo "cache\_run\_root=$CACHE\_RUN\_ROOT"
echo "target\_contract=$TARGET\_CONTRACT"
echo "channel\_name\_mapping=$CHANNEL\_NAME\_MAPPING"
echo "explicit\_mapping=CB1:PO9,CB2:PO10"
echo "log\_file=$LOG\_FILE"

set +e
(
set -Eeuo pipefail
cd "$HUADA\_ROOT"
```bash
"$PYTHON_BIN" -c "
```

import mne
import numpy
import scipy

print("python dependencies: OK")
print("mne=", mne.**version**)
print("numpy=", numpy.**version**)
print("scipy=", scipy.**version**)
"
```bash
"$PYTHON_BIN" -m scripts.native_backends.freeze_huada_target_contract \
    --model-family reve \
    --reve-position-source "$REVE_POSITIONS_SOURCE" \
    --expected-reve-position-source-sha256 "$EXPECTED_REVE_POSITIONS_SHA256" \
    --channel-name-mapping "$CHANNEL_NAME_MAPPING" \
    --output "$TARGET_CONTRACT"

test -s "$TARGET_CONTRACT"
test ! -L "$TARGET_CONTRACT"

"$PYTHON_BIN" -c "
```

import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
contract = json.load(handle)

assert contract["model\_family"] == "reve"
assert contract["status"] == "frozen"
assert contract["release\_status"] == "released"
assert contract["validation\_mode"] == "formal"
assert contract["position\_bank"]["position\_mask"] is None
assert contract["position\_bank"]["resolved\_position\_names"][60:62] == ["PO9", "PO10"]
assert contract["position\_bank"]["source\_asset\_sha256"] == sys.argv[2]

print("frozen REVE target contract: OK")
print("contract\_sha256=", contract["contract\_sha256"])
print(
"resolved\_CB\_positions=",
contract["position\_bank"]["resolved\_position\_names"][60:62],
)
" "$TARGET\_CONTRACT" "$EXPECTED\_REVE\_POSITIONS\_SHA256"
```bash
contract_sha_line="$(sha256sum "$TARGET_CONTRACT")"
TARGET_CONTRACT_FILE_SHA256="${contract_sha_line%% *}"

[[ "$TARGET_CONTRACT_FILE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: invalid target contract file SHA"
    exit 2
}

echo "target_contract_file_sha256=$TARGET_CONTRACT_FILE_SHA256"

for index in "${!SUBJECT_IDS[@]}"; do
    recording_id="${SUBJECT_IDS[$index]}"
    source_bdf="${SOURCE_BDFS[$index]}"
    output_dir="$RECORDING_CACHE_ROOT/$recording_id"

    echo "------------------------------------------------------------"
    echo "building_recording_id=$recording_id"
    echo "source_bdf=$source_bdf"
    echo "output_dir=$output_dir"

    "$PYTHON_BIN" -m scripts.native_backends.build_huada_native_recording_cache \
        --model-family reve \
        --source-bdf "$source_bdf" \
        --recording-id "$recording_id" \
        --target-input-contract "$TARGET_CONTRACT" \
        --expected-target-contract-file-sha256 "$TARGET_CONTRACT_FILE_SHA256" \
        --output-dir "$output_dir"

    test -s "$output_dir/recording.float32.npy"
    test -s "$output_dir/metadata.json"
    test -f "$output_dir/_SUCCESS"

    echo "recording_cache_passed=$recording_id"
done

for subject_id in "${SUBJECT_IDS[@]}"; do
    test -s "$RECORDING_CACHE_ROOT/$subject_id/recording.float32.npy"
    test -s "$RECORDING_CACHE_ROOT/$subject_id/metadata.json"
    test -f "$RECORDING_CACHE_ROOT/$subject_id/_SUCCESS"
done

printf "%s\n" \
    "model_family=reve" \
    "run_id=$RUN_ID" \
    "recording_count=${#SUBJECT_IDS[@]}" \
    "position_source=$REVE_POSITIONS_SOURCE" \
    "position_source_sha256=$EXPECTED_REVE_POSITIONS_SHA256" \
    "channel_mapping_CB1=PO9" \
    "channel_mapping_CB2=PO10" \
    "target_contract=$TARGET_CONTRACT" \
    "target_contract_file_sha256=$TARGET_CONTRACT_FILE_SHA256" \
    > "$CACHE_RUN_ROOT/_SUCCESS"

test -f "$CACHE_RUN_ROOT/_SUCCESS"

echo "============================================================"
echo "REVE native cache completed successfully"
echo "cache_run_root=$CACHE_RUN_ROOT"
echo "recording_count=${#SUBJECT_IDS[@]}"
echo "target_contract_file_sha256=$TARGET_CONTRACT_FILE_SHA256"
```

) 2>&1 | tee "$LOG\_FILE"

status=${PIPESTATUS[0]}
set -e

echo "exit\_code=$status"
echo "log\_file=$LOG\_FILE"
echo "cache\_run\_root=$CACHE\_RUN\_ROOT"

if [[ "$status" -ne 0 ]]; then
echo "REVE native cache failed; \_SUCCESS must be absent"
exit "$status"
fi

test -f "$CACHE\_RUN\_ROOT/\_SUCCESS"

echo "FINAL\_STATUS=SUCCESS"
echo "REVE\_CACHE\_ROOT=$CACHE\_RUN\_ROOT"
'
````

#### `train-1713905320562132736`：Dortmund backbone + age probe + source scaler 的华大 strict zero-shot

- 任务：`train-1713905320562132736`
- 产物：`/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_shot/huada_reve_dortmund_age_strict_zero_shot_20261109T094439Z`
- 命令来源：日志 `resolved_command=`

```bash
/opt/venv/bin/python -m scripts.native_backends.run_huada_portable_age_zero_shot --model-family reve --native-cache-run-root /mnt/cfs/jianyu/huada-eeg-pipeline/data/native_recording_cache/huada_reve/20261109T085249Z --subject-manifest /mnt/cfs/jianyu/huada-eeg-pipeline/preprocessing_audit/20261030_105520/subject_manifest.json --target-input-contract /mnt/cfs/jianyu/huada-eeg-pipeline/data/native_recording_cache/huada_reve/20261109T085249Z/huada_reve_target_input_contract_v1.json --expected-target-contract-file-sha256 8df337850379faac426040d2304a6a9245efbbe98facddbaa0e9b2fbcf702120 --portable-checkpoint /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_reve/20261109T052927Z/dortmund_age_reve_fft_portable_v1.pt --portable-sidecar /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_reve/20261109T052927Z/dortmund_age_reve_fft_portable_v1.sidecar.json --expected-portable-checkpoint-sha256 e2a7f72d435e7c8bc7688e7c27d0db5cccbc226b8dcf0acaf21172a38ed8fff4 --expected-source-checkpoint-sha256 ca3474382da2f03937232eedcdbaf7faf511725af2542561eeb5e7e0d5631201 --dynamic-parity-dir /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_parity/dortmund_age_reve/20261109T072416Z --output-dir /mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_shot/huada_reve_dortmund_age_strict_zero_shot_20261109T094439Z --device cuda:0 --batch-size 2 --expected-source-age-mean 27.5 --expected-source-age-std 4.218591213226318 --comparison-result fbmae=/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/50m_dortmund_age_huada_zero_shot_20261103_154351 --comparison-result labram=/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_shot/huada_labram_dortmund_age_strict_zero_shot_20261109T093949Z
```

#### `train-1714405767367109376`：相同 backbone/cache/CV7 的 age probe 零初始化训练

- 任务：`train-1714405767367109376`
- 产物：`/mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_init_probe_cv7/huada_reve_zero_init_probe_cv7_20261110T021852Z`
- feature cache：`/mnt/cfs/jianyu/huada-eeg-pipeline/data/native_feature_cache/huada_reve_portable_age_mean`
- 命令来源：日志 `resolved_command=`

```bash
/opt/venv/bin/python -m scripts.native_backends.run_huada_portable_age_zero_init_probe_cv7 --model-family reve --native-cache-run-root /mnt/cfs/jianyu/huada-eeg-pipeline/data/native_recording_cache/huada_reve/20261109T085249Z --subject-manifest /mnt/cfs/jianyu/huada-eeg-pipeline/preprocessing_audit/20261030_105520/subject_manifest.json --target-input-contract /mnt/cfs/jianyu/huada-eeg-pipeline/data/native_recording_cache/huada_reve/20261109T085249Z/huada_reve_target_input_contract_v1.json --expected-target-contract-file-sha256 8df337850379faac426040d2304a6a9245efbbe98facddbaa0e9b2fbcf702120 --portable-checkpoint /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_reve/20261109T052927Z/dortmund_age_reve_fft_portable_v1.pt --portable-sidecar /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_checkpoints/dortmund_age_reve/20261109T052927Z/dortmund_age_reve_fft_portable_v1.sidecar.json --expected-portable-checkpoint-sha256 e2a7f72d435e7c8bc7688e7c27d0db5cccbc226b8dcf0acaf21172a38ed8fff4 --expected-source-checkpoint-sha256 ca3474382da2f03937232eedcdbaf7faf511725af2542561eeb5e7e0d5631201 --expected-source-age-mean 27.5 --expected-source-age-std 4.218591213226318 --dynamic-parity-dir /mnt/cfs/jianyu/neuroai-repro/repro_heavy_local/portable_parity/dortmund_age_reve/20261109T072416Z --strict-zero-shot-reference-dir /mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_shot/huada_reve_dortmund_age_strict_zero_shot_20261109T094439Z --feature-cache-root /mnt/cfs/jianyu/huada-eeg-pipeline/data/native_feature_cache/huada_reve_portable_age_mean --feature-cache-policy auto --output-dir /mnt/cfs/jianyu/huada-eeg-pipeline/downstream_output/native_portable_age_zero_init_probe_cv7/huada_reve_zero_init_probe_cv7_20261110T021852Z --device cuda:0 --feature-cache-batch-size 8 --batch-size 64 --head-lr 1e-5 --weight-decay 0 --epochs 100 --early-stopping-patience 15 --seed 42 --split-seed 42
```

## 无有效 stdout 的成功任务

以下任务在 manifest 中状态为 `SUCCEED`，但下载归档中的日志文件为空，因此无法仅凭本目录判断实验内容、产物或入口命令：

- `train-1713750472462707456`
- `train-1713869689732213504`

## 失败任务

下载清单中有两项状态为 `FAILED`，下载器按规则跳过，本文不把它们列为成功流程，也不尝试恢复命令：

- `train-1706671442327897856`
- `train-1706627563731148672`

## 日志定位规则

每个已下载任务的原始日志归档位于：

```text
train-<task-id>/log_<task-id>_..._<export-id>.tar.gz
```

文中任务状态以 `download_manifest_20261110_124317_093.json` 为准；实验含义和产物路径以对应归档中的 stdout/stderr 为准。

## 目前实验结果(age)
![FBMAE+labram+reve](downstream_output\age_model_comparison_20261110\age_prediction_vs_ground_truth.png)

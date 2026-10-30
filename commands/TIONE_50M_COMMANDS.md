# Ti-ONE 上运行 50M 性别 downstream

以下命令均为 Linux/Bash。正式协议为 `sex_stratified_subject_cv4`，完整流程按顺序完成被试切分、4 折训练、每折最佳 checkpoint 选择、test evaluation、OOF 合并和被试级聚合。

GPU 型号和数量在 Ti-ONE 任务或开发机界面选择。代码、正式 Lance、预训练 checkpoint 和输出目录应通过 CFS/COSFS 挂载或平台环境变量传入；不要把腾讯云凭据写入代码。

## 1. 公共环境变量和运行前检查

先按 Ti-ONE 的实际挂载路径填写：

```bash
set -euo pipefail

export HUADA_ROOT="/workspace/huada"
export PYTHON_BIN="python"

# 正式 model-token Lance。可以是挂载目录，也可以是 Lance 远程 URI。
export INPUT_LANCE="/mnt/huada/model_tokens_full/huada_model_tokens.lance"

# 只有直接读取远程 URI 时才填写由 Ti-ONE 注入或挂载的 JSON 文件路径。
export STORAGE_OPTIONS=""

export PRETRAINED_50M_CKPT="/mnt/huada/checkpoints/50m_pretrained.pt"
export TIONE_OUTPUT_ROOT="/mnt/huada/downstream_outputs"

cd "$HUADA_ROOT"
test -d "$HUADA_ROOT"
test -f "scripts/downstream_50m/largeScale_fineTune.py"
test -f "commands/run_50m_sex_cv4_full.sh"
test -f "$PRETRAINED_50M_CKPT"
test -d "$TIONE_OUTPUT_ROOT"
if [[ "$INPUT_LANCE" != *"://"* ]]; then
  test -d "$INPUT_LANCE"
fi

LANCE_AUTH=()
if [[ -n "${STORAGE_OPTIONS:-}" ]]; then
  test -f "$STORAGE_OPTIONS"
  LANCE_AUTH=(--storage-options "$STORAGE_OPTIONS")
fi

"$PYTHON_BIN" -c 'import torch,lance,pyarrow; print({"torch":torch.__version__,"cuda":torch.cuda.is_available(),"lance":lance.__version__,"pyarrow":pyarrow.__version__})'
```

`INPUT_LANCE` 必须是正式 model-token Lance，不是 common-window Lance。正式运行还必须使用真实 50M pretrained backbone checkpoint。

## 2. Subject split 阶段

CV4 split 由正式训练入口在任何 fold 开始前自动生成，不需要另外运行旧的单次 split manifest 脚本。

固定规则：

```text
4 female + 4 male
→ female/male subject_id 分别排序并按 split seed 独立确定性打乱
→ 配成 4 个 pair，每个 pair 为 1 female + 1 male

fold i:
  test       = pair_i
  validation = pair_(i+1 mod 4)
  train      = 剩余两个 pair
```

默认 `SPLIT_SEED=42`。训练入口会先校验：

- 每折 train/validation/test 被试集合不相交；
- train 为 2 female + 2 male；
- validation 和 test 均为 1 female + 1 male；
- 每名被试在 4 折中恰好作为 test 被试一次；
- 同一被试的全部窗口只属于一个 split。

切分文件写入本次运行输出根目录：

```text
sex_cv4_subject_splits.csv
sex_cv4_subject_splits.json
```

## 3. 正式训练：masked mean（推荐）

```bash
POOLING=mean \
HEAD_LR=1e-3 \
WEIGHT_DECAY=1e-3 \
TRAIN_SEED=42 \
SPLIT_SEED=42 \
EPOCHS=5 \
BATCH_SIZE=64 \
bash commands/run_50m_sex_cv4_full.sh
```

对应 head：

```text
head_input_dim = 512
head_parameter_count = 1026
```

## 4. 正式训练：masked flatten（对照）

```bash
POOLING=flatten \
HEAD_LR=1e-4 \
WEIGHT_DECAY=1e-3 \
TRAIN_SEED=42 \
SPLIT_SEED=42 \
EPOCHS=5 \
BATCH_SIZE=64 \
bash commands/run_50m_sex_cv4_full.sh
```

对应 head：

```text
head_input_dim = 640 × 512 = 327680
head_parameter_count = 655362
```

每次脚本调用只执行一个配置，不会自动运行参数网格。输出目录包含 pooling、head learning rate、weight decay 和时间戳，例如：

```text
50m_sex_cv4_mean_lr1e-3_wd1e-3_YYYYmmdd_HHMMSS
50m_sex_cv4_flatten_lr1e-4_wd1e-3_YYYYmmdd_HHMMSS
```

## 5. 每折 train 和 checkpoint 选择

脚本内部只调用一次正式入口；该入口依次运行 4 个 fold。每折都会：

1. 使用 `fold_seed = TRAIN_SEED + fold`；
2. 新建 50M backbone 和 downstream head；
3. 从同一个 pretrained checkpoint 重新加载 backbone；
4. 新建 optimizer，不复用上一折状态；
5. 只使用 train subjects 训练；
6. 只使用 validation subjects 选择最佳 checkpoint；
7. 把最佳 checkpoint 保存到独立 fold 目录。

目录结构：

```text
fold_0/checkpoints/downstream_finetune.pt
fold_1/checkpoints/downstream_finetune.pt
fold_2/checkpoints/downstream_finetune.pt
fold_3/checkpoints/downstream_finetune.pt
```

linear probe 保持使用 SGD；finetune 模式保持使用 AdamW。本轮正式示例使用 `--mode linear_probe`。`--head-lr` 和 `--weight-decay` 会显式传入，并从真实 optimizer param groups 记录到 checkpoint、fold result 和 provenance。

## 6. Evaluation 和 OOF 聚合

CV4 evaluation 是正式入口的一部分，不应在训练结束后再启动一套独立 test 命令。每折流程固定为：

```text
validation 选择最佳 checkpoint
→ 重新加载该 fold 最佳 checkpoint
→ test pair 只评估一次
→ 保存每个测试窗口的 softmax 概率
```

4 折完成后，入口自动合并 test-only OOF 预测，并重新计算：

- window-level accuracy、balanced accuracy、macro-F1、confusion matrix；
- subject-level probability mean 指标；
- subject-level majority vote 指标及 tie-break；
- per-fold test 指标。

正式输出应包含：

```text
sex_cv4_subject_splits.csv
sex_cv4_subject_splits.json
sex_cv4_oof_window_predictions.csv
sex_cv4_oof_subject_predictions.csv
sex_cv4_per_fold_metrics.csv
sex_cv4_window_metrics.json
sex_cv4_subject_metrics.json
sex_cv4_summary.json
sex_cv4_report.md
config.json
processing_provenance.json
summary.json
```

## 7. 运行后检查

脚本结束时会打印：

```text
CV4 output: <本次输出目录>
```

进入该目录后检查：

```bash
cd "<REPLACE_WITH_CV4_OUTPUT_DIR>"

test -f sex_cv4_subject_splits.csv
test -f sex_cv4_oof_window_predictions.csv
test -f sex_cv4_oof_subject_predictions.csv
test -f sex_cv4_window_metrics.json
test -f sex_cv4_subject_metrics.json
test -f sex_cv4_summary.json

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("sex_cv4_summary.json").read_text(encoding="utf-8"))
print(json.dumps({
    "protocol": summary["protocol"],
    "fold_count": summary["fold_count"],
    "pooling": summary["pooling"],
    "head_lr": summary["head_lr"],
    "weight_decay": summary["weight_decay"],
    "head_input_dim": summary["head_input_dim"],
    "head_parameter_count": summary["head_parameter_count"],
    "window_metrics": summary["window_metrics"],
    "subject_metrics": summary["subject_metrics"],
}, ensure_ascii=False, indent=2))
PY
```

window-level 指标以窗口为统计单位；subject-level 指标以 8 名被试为统计单位。subject-level 是本任务更重要的结果，但 8 名被试仍属于极小样本 pilot，不能声称具有稳定总体泛化性能。

## 8. 正式入口的等价展开命令

通常直接使用 `commands/run_50m_sex_cv4_full.sh`。为了便于在 Ti-ONE 控制台核对，脚本中的唯一训练调用等价于：

```bash
"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py \
  --input-lance "$INPUT_LANCE" \
  "${LANCE_AUTH[@]}" \
  --task sex_classification \
  --protocol sex_stratified_subject_cv4 \
  --folds 4 \
  --pooling "$POOLING" \
  --head-lr "$HEAD_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint "$PRETRAINED_50M_CKPT" \
  --device cuda \
  --mode linear_probe \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$TRAIN_SEED" \
  --split-seed "$SPLIT_SEED"
```

不要加入 `--smoke`、样本数限制、随机初始化或 `--overwrite`。正式 CV4 只在 Ti-ONE 运行。

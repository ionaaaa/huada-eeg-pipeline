# Ti-ONE 上运行 50M downstream

以下命令均为 Linux/Bash。GPU 型号和数量在 Ti-ONE 任务/开发机界面选择；不要在命令中写腾讯云密钥。代码、checkpoint、正式 Lance 和输出目录应通过 CFS/COSFS 挂载，或由平台环境变量传入。

## 1. 公共环境变量

先按 Ti-ONE 实际挂载路径填写：

```bash
set -euo pipefail

export HUADA_ROOT="/workspace/huada"
export PYTHON_BIN="python"

# 二选一：
# A. 推荐：COS/CFS 已挂载到容器，本变量填写本地 Lance 目录。
export INPUT_LANCE="/mnt/huada/model_tokens_full/huada_model_tokens.lance"
# B. 直接使用 Lance 远程 URI 时，INPUT_LANCE 改成对应 URI，并填写 storage options JSON。
export STORAGE_OPTIONS=""

export PRETRAINED_50M_CKPT="/mnt/huada/checkpoints/50m_pretrained.pt"
export TIONE_OUTPUT_ROOT="/mnt/huada/downstream_outputs"

cd "$HUADA_ROOT"
test -f "$PRETRAINED_50M_CKPT"
test -d "$INPUT_LANCE" || [[ "$INPUT_LANCE" == *"://"* ]]
mkdir -p "$TIONE_OUTPUT_ROOT"

LANCE_AUTH=()
if [[ -n "${STORAGE_OPTIONS:-}" ]]; then
  test -f "$STORAGE_OPTIONS"
  LANCE_AUTH=(--storage-options "$STORAGE_OPTIONS")
fi

"$PYTHON_BIN" -c 'import torch,lance,pyarrow; print({"torch":torch.__version__,"cuda":torch.cuda.is_available(),"lance":lance.__version__,"pyarrow":pyarrow.__version__})'
```

`INPUT_LANCE` 是已生成的正式 model-token Lance，不是 common-window Lance。若使用挂载目录，训练进程不会读取云密钥；若使用远程 URI，`STORAGE_OPTIONS` 应指向 Ti-ONE 注入或挂载的 JSON，禁止提交到 Git。

## 推荐：性别 4 折被试级交叉验证

新的正式性别协议为 `sex_stratified_subject_cv4`。完成上面的公共环境变量设置后直接运行：

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"
SEX_CV4_OUTPUT="$TIONE_OUTPUT_ROOT/50m_sex_cv4_$STAMP"
test ! -e "$SEX_CV4_OUTPUT"

"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py \
  --input-lance "$INPUT_LANCE" \
  "${LANCE_AUTH[@]}" \
  --task sex_classification \
  --protocol sex_stratified_subject_cv4 \
  --folds 4 \
  --output-dir "$SEX_CV4_OUTPUT" \
  --checkpoint "$PRETRAINED_50M_CKPT" \
  --device cuda \
  --mode linear_probe \
  --epochs 5 \
  --batch-size 64 \
  --seed 42 \
  --split-seed 42

echo "CV4 output: $SEX_CV4_OUTPUT"
```

命令会在正式 Lance 中核验恰好 4 female + 4 male，并生成每折 manifest、独立 checkpoint、test-only OOF 窗口预测和被试聚合指标。它不包含 smoke、样本限制、随机初始化或覆盖参数。

下述单次 split 性别命令仅为历史复现，协议标记为 `legacy_single_split`，不再作为推荐正式协议。

## 2. 生成性别 manifest/split（legacy_single_split）

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"
SEX_MANIFEST_DIR="$TIONE_OUTPUT_ROOT/manifests/sex_$STAMP"

"$PYTHON_BIN" scripts/create_downstream_manifest.py \
  --input-lance "$INPUT_LANCE" \
  "${LANCE_AUTH[@]}" \
  --task sex_classification \
  --seed 42 \
  --output-manifest "$SEX_MANIFEST_DIR/sex_subject_manifest.csv" \
  --output-splits "$SEX_MANIFEST_DIR/sex_subject_splits.csv"

export SEX_SPLIT_MANIFEST="$SEX_MANIFEST_DIR/sex_subject_splits.csv"
```

## 3. 生成年龄 manifest/split

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"
AGE_MANIFEST_DIR="$TIONE_OUTPUT_ROOT/manifests/age_$STAMP"

"$PYTHON_BIN" scripts/create_downstream_manifest.py \
  --input-lance "$INPUT_LANCE" \
  "${LANCE_AUTH[@]}" \
  --task age_regression \
  --seed 42 \
  --output-manifest "$AGE_MANIFEST_DIR/age_subject_manifest.csv" \
  --output-splits "$AGE_MANIFEST_DIR/age_subject_splits.csv"

export AGE_SPLIT_MANIFEST="$AGE_MANIFEST_DIR/age_subject_splits.csv"
```

生成后先检查：

```bash
cat "$SEX_SPLIT_MANIFEST"
cat "$AGE_SPLIT_MANIFEST"
```

年龄 manifest 中 SUB5 应显示 `age_missing` 且 split 为空。

## 4. 旧版单次 split 性别训练（legacy_single_split）

```bash
: "${SEX_SPLIT_MANIFEST:?先生成并检查性别 split manifest}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SEX_OUTPUT="$TIONE_OUTPUT_ROOT/50m_sex_full_$STAMP"

"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py \
  --input-lance "$INPUT_LANCE" \
  "${LANCE_AUTH[@]}" \
  --task sex_classification \
  --protocol legacy_single_split \
  --subject-split-manifest "$SEX_SPLIT_MANIFEST" \
  --output-dir "$SEX_OUTPUT" \
  --checkpoint "$PRETRAINED_50M_CKPT" \
  --device cuda \
  --mode linear_probe \
  --epochs 5 \
  --batch-size 64 \
  --seed 42 \
  --split-seed 42

export SEX_DOWNSTREAM_CKPT="$SEX_OUTPUT/checkpoints/downstream_finetune.pt"
```

## 5. 正式年龄训练

```bash
: "${AGE_SPLIT_MANIFEST:?先生成并检查年龄 split manifest}"
STAMP="$(date +%Y%m%d_%H%M%S)"
AGE_OUTPUT="$TIONE_OUTPUT_ROOT/50m_age_full_$STAMP"

"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py \
  --input-lance "$INPUT_LANCE" \
  "${LANCE_AUTH[@]}" \
  --task age_regression \
  --subject-split-manifest "$AGE_SPLIT_MANIFEST" \
  --output-dir "$AGE_OUTPUT" \
  --checkpoint "$PRETRAINED_50M_CKPT" \
  --device cuda \
  --mode linear_probe \
  --epochs 5 \
  --batch-size 64 \
  --seed 42 \
  --split-seed 42

export AGE_DOWNSTREAM_CKPT="$AGE_OUTPUT/checkpoints/downstream_finetune.pt"
```

## 6. 旧版单次 split 性别 evaluation（legacy_single_split）

```bash
: "${SEX_DOWNSTREAM_CKPT:?填写性别 downstream checkpoint}"
STAMP="$(date +%Y%m%d_%H%M%S)"

"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py \
  --input-lance "$INPUT_LANCE" \
  "${LANCE_AUTH[@]}" \
  --task sex_classification \
  --protocol legacy_single_split \
  --subject-split-manifest "$SEX_SPLIT_MANIFEST" \
  --output-dir "$TIONE_OUTPUT_ROOT/eval_50m_sex_$STAMP" \
  --downstream-checkpoint "$SEX_DOWNSTREAM_CKPT" \
  --device cuda \
  --mode linear_probe \
  --evaluate
```

## 7. 年龄 evaluation

```bash
: "${AGE_DOWNSTREAM_CKPT:?填写年龄 downstream checkpoint}"
STAMP="$(date +%Y%m%d_%H%M%S)"

"$PYTHON_BIN" scripts/downstream_50m/largeScale_fineTune.py \
  --input-lance "$INPUT_LANCE" \
  "${LANCE_AUTH[@]}" \
  --task age_regression \
  --subject-split-manifest "$AGE_SPLIT_MANIFEST" \
  --output-dir "$TIONE_OUTPUT_ROOT/eval_50m_age_$STAMP" \
  --downstream-checkpoint "$AGE_DOWNSTREAM_CKPT" \
  --device cuda \
  --mode linear_probe \
  --evaluate
```

## 8. Ti-ONE 任务启动命令示例

若把上述变量写入 Ti-ONE 的环境变量配置，并把代码挂载到 `$HUADA_ROOT`，任务启动命令应依次执行“公共环境变量”代码块中的检查和 `LANCE_AUTH` 初始化，再执行“推荐：性别 4 折被试级交叉验证”代码块；不需要仓库内额外的 `.sh` 文件。

如果直连远程 Lance 而不是使用 COSFS/CFS 本地挂载，保留公共环境变量代码块中的：

```bash
--storage-options "$STORAGE_OPTIONS"
```

正式命令不得加入 `--smoke`、`--smoke-max-samples`、`--allow-random-init-smoke` 或 `--overwrite`。

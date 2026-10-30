# Ti-ONE：50M 性别被试级 CV4

正式云端协议固定为 `sex_stratified_subject_cv4`。脚本只执行环境检查、依赖版本打印和一次正式训练入口；不会生成旧 split manifest，也不会继续执行额外训练或评估。

## 环境变量

按 Ti-ONE 的实际挂载路径设置：

```bash
export HUADA_ROOT="/workspace/huada"
export INPUT_LANCE="/mnt/huada/model_tokens_full/huada_model_tokens.lance"
export PRETRAINED_50M_CKPT="/mnt/huada/checkpoints/50m_pretrained.pt"
export TIONE_OUTPUT_ROOT="/mnt/huada/downstream_outputs"
export PYTHON_BIN="python"

# 仅在直接读取远程 Lance URI 时填写由 Ti-ONE 注入或挂载的 JSON 路径。
export STORAGE_OPTIONS=""
```

`INPUT_LANCE` 必须是正式 model-token Lance，不是 common-window Lance。不要把腾讯云凭据写进代码或提交到 Git。

## 推荐：masked mean

```bash
POOLING=mean \
HEAD_LR=1e-3 \
WEIGHT_DECAY=1e-3 \
bash commands/run_50m_sex_cv4_full.sh
```

## 对照：masked flatten

```bash
POOLING=flatten \
HEAD_LR=1e-4 \
WEIGHT_DECAY=1e-3 \
bash commands/run_50m_sex_cv4_full.sh
```

脚本会显式传入 `--pooling`、`--head-lr` 和 `--weight-decay`。输出目录包含 pooling、学习率、weight decay 与时间戳，例如：

```text
50m_sex_cv4_mean_lr1e-3_wd1e-3_YYYYmmdd_HHMMSS
50m_sex_cv4_flatten_lr1e-4_wd1e-3_YYYYmmdd_HHMMSS
```

脚本不会自动运行参数网格。每次调用只运行一个配置，且输出目录已存在时会停止。

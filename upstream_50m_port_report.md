# 50M downstream 迁移报告

## 审计基线

- 上游：`https://github.com/ionaaaa/50M_Model`
- branch：`main`
- commit：`0f102b10ac885df759f1d7052a36669c7412c5e6`
- 主要入口：`largeScale_fineTune.py`
- 其他已审计入口：`largeScale_fineTune_ddp_bacc.py`、`largeScale_fineTune_ddp_clean.py`、`evaluate_best_ckpts_bacc.py`
- dataset/dataloader：`lance_dataset.py` 的 `LanceEEGPretrainDataset` / `MultiLanceEEGPretrainDataset`，`dataset.py::collate_fn`，可选 `PrefetchLoader`
- backbone：`model.py::EEGPretrainModel`，50M 配置为 `d_model=512, n_heads=8, depth=12`
- downstream head：`EEGDownstreamClassifier`，默认单层线性 head；默认不用 MLP
- representation：`probe_layer_ratio=0.75`，12 层时为 0-based 第 8 层（第 9 层）输出
- pooling：实际代码为先乘 `token_valid_mask` 再 flatten；上游 docstring 中的 “valid token mean pooling” 与实际实现不一致
- optimizer：linear probe 为 SGD；finetune 为 AdamW，backbone/head 可用不同学习率
- scheduler：无
- 分类 loss：cross entropy
- 上游指标：loss、accuracy；DDP bacc 版本另有 balanced accuracy
- 流程：每 epoch train/validation，按 val metric 保存最佳模型，最后在 test 上评估
- checkpoint loader：依次读取 `model_state_dict`、`state_dict` 或顶层 dict，剥离 `module.`，`strict=False`
- 保存格式：完整 model/head/backbone state dict、配置、shape、epoch 和指标
- 恢复训练：主要入口没有 optimizer/scheduler resume；evaluation 脚本读取已训练 checkpoint

## 文件迁移与必要改动

详细逐文件表见 `upstream_50m_port_manifest.json`。其中 `model.py`、`prefetch_loader.py`、`utils.py` 的 SHA256 与上游完全一致。

| 上游文件 | 当前项目文件 | 处理方式 | 修改原因 |
|---|---|---|---|
| `model.py` | `scripts/downstream_50m/model.py` | 原样复制 | 保持 50M backbone、representation API 和预训练 head 完全一致 |
| `prefetch_loader.py` | `scripts/downstream_50m/prefetch_loader.py` | 原样复制 | 保持 loader 行为 |
| `utils.py` | `scripts/downstream_50m/utils.py` | 原样复制 | 保持 seed 行为 |
| `dataset.py` | `scripts/downstream_50m/dataset.py` | 少量修改 | collate 增加字符串 metadata 与采样率；tensor stacking 未改 |
| `configs.py` | `scripts/downstream_50m/configs.py` | 少量修改 | 去掉上游硬编码云路径，增加 task/manifest/checkpoint/smoke 参数；模型/优化器默认不改 |
| `lance_dataset.py` | `scripts/downstream_50m/lance_dataset.py` | 少量修改 | 从共享 Lance 的 `sex`/`age` 动态生成 label；按 manifest 过滤；不复制 token |
| `largeScale_fineTune.py` | `scripts/downstream_50m/largeScale_fineTune.py` | 必要修改 | 修复 subject 泄漏；增加年龄回归、完整指标、CLI、checkpoint 审计、smoke/eval |
| 无 | `scripts/create_downstream_manifest.py` | 新增薄适配器 | 生成 task manifest 和 subject split |

## 上游发现的问题

1. 上游 `cross_subject` 只是按样本 index 顺序切分，并假设被试连续；切分边界可以落在同一被试内部。这不是可靠的 subject-level split。本迁移只替换 split 入口，要求显式 subject manifest，并验证集合两两不相交。
2. `EEGDownstreamClassifier` 注释称 valid-token mean pooling，但 forward 实际使用 masked flatten。本迁移保留实际数学行为，报告中不把它描述为 mean pooling。
3. checkpoint loader 使用 `strict=False`，上游只打印 mismatch。本迁移仍复用 key 选择和 prefix 清理，但把 missing/unexpected 写入 provenance，超过 8 个 mismatch 时失败，避免静默跳过大量参数。
4. 上游只有分类。本迁移用同一个 head 构造（输出维度 1）、同一个训练循环和 MSELoss 增加最薄回归分支；回归输出在 loss 前统一为 `[batch]`。

## 实际 Lance 契约

本地 smoke Lance 已验证为 4 行，核心字段和 shape 与预期一致：`token_inputs=[640,100]`、`targets=[640,5,5]`、两个 index/mask 均为 640、`channel_valid_mask=64`，`sample_id/subject_id/recording_id/age/sex/source_sfreq/model_sfreq` 均存在。

实际 montage 存在显式差异：模型 64 通道表包含 `AFz/Iz/F9/F10`，华大源 montage 缺少这些通道，因此普通被试是 60 个有效通道/600 个有效 token；SUB5 再因 FC1 坏道成为 59/590。缓存没有被改写或伪装成 64/640 与 63/630。测试同时保留了通用的 64/640、单坏道 63/630 fixture，并验证 SUB8 的 FT7 对应 10 个连续 token 被 mask。

mask 在 backbone 后、上游 flatten 前生效：token embedding 先乘 `token_valid_mask[..., None]`，无效 token 的整段 embedding 变成零，再进入线性 head。backbone 本身没有 attention mask，因此无效零 token 仍会经过 encoder；本轮按要求不修改 backbone 架构。

## 标签与 split

- 性别：显式常量 `female=0, male=1`，输出/manifest/provenance 均记录映射，label 为 int64。SUB5 保留。
- 年龄：直接读取 nullable `age`，label 为 float32；SUB5 的 null 被标记为 `age_missing` 并排除，不填 0、不插值、不读取 BDF 占位年龄。
- 正式 split：`manifests/*_subject_splits.csv` 以 subject 为唯一分配单位，seed 42；性别为 6/1/1 subject，年龄为 5/1/1 eligible subject。这只是可复现的机械划分，不代表统计上最优或有代表性。
- smoke：因性别只有 2 个被试、年龄只有 1 个 eligible 被试，不构造假的三分 split；全部 eligible smoke 被试仅标记为 train，validation/test 为空且不发生交叉。

## checkpoint 与 parity

未找到并未加载可确认属于当前上游 50M 架构的 pretrained checkpoint。`D:\repo\xxxxxxxx\checkpoints\pretrain_real_set_best.pt` 属于本地旧版/不同 50M 配置（256×6），没有冒充当前 512×12 checkpoint。本轮 smoke 使用显式 `--allow-random-init-smoke`，provenance 中 `random_init=true`。

parity test 已通过：固定 seed、相同输入、相同权重，对比上游 masked-flatten/head 路径与迁移版本，`atol=rtol=1e-6`。由于 pretrained checkpoint 不可用，未执行真实 pretrained checkpoint 端到端 parity，不能宣称 checkpoint 级完全等价。

## 正式运行阻塞

正式 model-token Lance 位于腾讯云，本轮没有下载、搜索本地副本或修改它。`commands/` 中保留远程 Lance URI、storage options JSON 和 50M pretrained checkpoint 三个显式占位符；填入后才可运行正式 manifest、训练与 evaluation。命令不含 smoke、样本限制、random-init 或 overwrite 参数。

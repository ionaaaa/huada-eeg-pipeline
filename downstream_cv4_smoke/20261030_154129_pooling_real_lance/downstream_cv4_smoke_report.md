# 50M 性别 downstream CV4 smoke 报告

- 时间：2026-10-30 15:41:29 +08:00
- 真实输入：`D:\repo\华大\model_tokens_smoke\20261030_134527\huada_model_tokens.lance`
- 真实数据 smoke：2 个 SUB1 窗口，CPU，1 个机械 train step
- pooling：masked mean
- head：`512 -> 2`，1026 parameters
- 输入快照前后一致：true
- 输出：`D:\repo\华大\downstream_cv4_smoke\20261030_154129_pooling_real_lance`

实际命令：

```powershell
& 'D:\miniconda\python.exe' 'D:\repo\华大\scripts\downstream_50m\largeScale_fineTune.py' `
  --input-lance 'D:\repo\华大\model_tokens_smoke\20261030_134527\huada_model_tokens.lance' `
  --task sex_classification --protocol legacy_single_split `
  --output-dir 'D:\repo\华大\downstream_cv4_smoke\20261030_154129_pooling_real_lance' `
  --device cpu --mode linear_probe --epochs 1 --batch-size 2 `
  --seed 42 --split-seed 42 --smoke --smoke-max-samples 2 `
  --allow-random-init-smoke
```

完整 4 折编排 smoke 由 `test_cv4_orchestration_smoke_writes_fold_checkpoints_and_oof`
使用 8 被试机械 fixture 验证：4 个独立 fold 目录/checkpoint、test-only OOF、全局窗口指标和
被试级聚合均写出。fixture 只验证控制流，不代表 EEG 模型性能。

本地真实 token 仅覆盖 SUB1、SUB5，且两者均为 male；本地也没有真实 pretrained 50M
checkpoint。因此没有伪造 female 标签或执行不合规的真实 4 折 smoke。上面的 accuracy 仅来自
单类别、随机初始化、2 窗口机械步骤，`metrics_not_statistically_meaningful=true`，不得作为性能结果。


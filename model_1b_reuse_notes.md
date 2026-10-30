# 1B 最小复用说明

本轮没有实现 1B downstream。

## 可直接复用

- `lance_dataset.py` 的共享 Lance 读取、动态 task label、SUB5 年龄排除
- task/split manifest、subject 集合不相交与窗口继承验证
- sex/age loss 与指标逻辑
- dataloader、provenance、输入保护、non-overwrite、smoke/full CLI 约束
- train/validation/test 循环与 checkpoint 输出结构的大部分字段

## 输入 contract

当前生成器保存的 `model_contract_50m.json` 与 `model_contract_1b.json` 显示数据几何相同：100 Hz、64 通道、10 个 1 秒 patch/通道、640 token、每 token 100 samples、target 为 5×5。因此共享 Lance 无需复制，dataset/task/split 无需变化。

实际华大通道缺失（AFz/Iz/F9/F10）也会同样影响 1B；仍应消费同一个 `token_valid_mask`。

## 需要替换的最小点

1. 新增一个 1B model factory 或直接在独立 1B 入口中构造其 `EEGPretrainModel`；不要提前增加通用注册框架。
2. 把 50M 的 `d_model=512, n_heads=8, depth=12` 换成经目标 1B checkpoint 验证的几何。公共 contract 当前记录为 `d_model=2048, n_heads=16, depth=20`，head 输入维度必须随 d_model 变化。
3. 使用 1B 自己的 checkpoint loader/key 规范并做 geometry 检查。不能复用 50M checkpoint，也不能只靠 `strict=False` 忽略差异。
4. 明确 1B representation layer 与 pooling。不要自动沿用 50M 的第 9 层 masked flatten；应以 1B 上游实际 downstream 为准。如果使用 masked mean，分母必须排除无效 token。
5. 为显存单独配置 batch size、混合精度和梯度累积；这些是运行参数，不要求改 dataset/trainer 抽象。1B full finetune 很可能需要更小 batch、AMP/bfloat16 和 accumulation，linear probe 可先冻结并缓存 feature。

## 本地 checkout 差异

本地 `D:\repo\EEG_pretrain` 为 `main@42ab6553e49cea281b6bd651790b1dc3540726ee`，现有 `largeScale_linearProbe_5b.py` 实际写死 4096×25 的 5B 几何、最后层 masked mean、float16 feature cache，并非上述 1B 2048×20 配置。因此它可用于理解大模型 checkpoint geometry/AMP/cache 策略，但不能直接当作 1B model factory。

## 最少修改文件

- 新增：一个独立 `scripts/downstream_1b/...` model/entry（从 1B 上游复制后最小适配）
- 复用不改：`scripts/downstream_50m/lance_dataset.py`、`dataset.py`、`scripts/create_downstream_manifest.py`、manifests
- 小改/参数化：head 的 feature dimension、checkpoint provenance 字段、1B 专属 batch/AMP/accumulation 配置

在获得确切 1B checkpoint 和对应上游 commit 前，不应实现或声称 checkpoint 可加载。

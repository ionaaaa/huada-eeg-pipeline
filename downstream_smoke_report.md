# 50M downstream smoke 报告

## 输入与保护

- 输入：`D:\repo\华大\model_tokens_smoke\20261030_134527\huada_model_tokens.lance`
- 行数：4（SUB1 两窗，SUB5 两窗）
- Lance 文件摘要（两次运行前后均相同）：`8a76a1a63e4c0aedf7019d5e83b61eaaa04381f4dddb0aa23be37dacea6fe286`
- 未读取 BDF、未重新滤波/重采样/切窗/归一化/计算 token，未修改 `targets`
- 两次均使用 50M 512×12 backbone、默认第 9 层 representation、masked flatten、linear-probe SGD
- 两次均为显式随机初始化 smoke；没有验证 pretrained checkpoint 加载

## 性别 smoke

- 输出：`D:\repo\华大\downstream_smoke\20261030_143117_sex`
- 可用数据：4 行、2 被试（SUB1、SUB5）；实际单步 forward/backward：2 行
- 标签：两者均为 male=1；SUB5 被保留
- loss：0.2320944369
- accuracy：1.0
- balanced accuracy：不计算（单一真实类别）
- macro-F1：不计算（单一真实类别）
- confusion matrix：`[[0,0],[0,2]]`
- `metrics_not_statistically_meaningful=true`
- checkpoint SHA256：`ecc17b5886932e9b179a615aaa07d6a69bc4e4d3a867e5d09e8e25a512e0e29e`

实际命令：

```powershell
D:\miniconda\python.exe D:\repo\华大\scripts\downstream_50m\largeScale_fineTune.py --input-lance D:\repo\华大\model_tokens_smoke\20261030_134527\huada_model_tokens.lance --task sex_classification --output-dir D:\repo\华大\downstream_smoke\20261030_143117_sex --device cuda --mode linear_probe --epochs 1 --batch-size 2 --smoke --smoke-max-samples 4 --allow-random-init-smoke
```

## 年龄 smoke

- 输出：`D:\repo\华大\downstream_smoke\20261030_143135_age`
- 可用数据：2 行、1 被试（SUB1）；实际单步 forward/backward：2 行
- SUB5：因 `age_missing` 明确排除
- loss/MSE：892.7751464844
- MAE：29.8793416023
- RMSE：29.8793426871
- R²：不计算（只有一个被试且真实年龄为常数）
- `metrics_not_statistically_meaningful=true`
- checkpoint SHA256：`84e33136ad00edf93711adb48ab5fd9647743e5d6b18b8e8189c942689678f81`

实际命令：

```powershell
D:\miniconda\python.exe D:\repo\华大\scripts\downstream_50m\largeScale_fineTune.py --input-lance D:\repo\华大\model_tokens_smoke\20261030_134527\huada_model_tokens.lance --task age_regression --output-dir D:\repo\华大\downstream_smoke\20261030_143135_age --device cuda --mode linear_probe --epochs 1 --batch-size 2 --smoke --smoke-max-samples 4 --allow-random-init-smoke
```

## 结论

两个任务都完成真实 Lance 读取、50M forward、loss backward、checkpoint/metrics/provenance 写出和输入不变校验。这里的数值只证明管线可机械运行：随机初始化、样本极少、性别单类、年龄单被试，因此不能作为任何模型性能结论。

# 华大静息态 EEG：阶段 1–3 验证报告

生成时间：2026-10-30T10:55:23+08:00  
原始数据目录：`D:\repo\华大\data\resting_state`  
配置：`D:\repo\华大\configs\huada_resting_preprocessing.toml`  
结果目录：`D:\repo\华大\preprocessing_audit\20261030_105520`

## 范围

本次只完成：固定并校验公共配置、建立目录名权威被试清单、执行原始 BDF 文件级硬性检查。未执行滤波、陷波、重采样、切窗、归一化、坏道修改、标签导出或 Lance 转换。原始数据未被修改。

## 公共配置

- 配置一致性校验：通过。
- `segment_samples == target_sfreq × window_seconds`：通过（2000 == 200 × 10）。
- `hop_seconds > 0`、`bandpass_low > 0`、`bandpass_high < target_sfreq/2`、`notch_hz < target_sfreq/2`：全部通过。
- 配置当前只保存和校验，没有执行预处理。

## 权威被试清单

- manifest 被试数：8。
- 人口学权威来源固定为 `directory_name`；BDF header 不覆盖 manifest。

- SUB1：age=31，sex=male，source=`SUB1_31_M`，status=complete
- SUB2：age=29，sex=female，source=`SUB2_29_F`，status=complete
- SUB3：age=26，sex=female，source=`SUB3_26_F`，status=complete
- SUB4：age=26，sex=female，source=`SUB4_26_F`，status=complete
- SUB5：age=null，sex=male，source=`SUB5_M`，status=age_missing
- SUB6：age=28，sex=male，source=`SUB6_28_M`，status=complete
- SUB7：age=28，sex=female，source=`SUB7_28_F`，status=complete
- SUB8：age=22，sex=male，source=`SUB8_22_M`，status=complete

## BDF 文件级硬性检查

- recording 数：8。
- 通过：8；失败：0。
- 参考通道顺序文件：`SUB1_31_M/BGI_EEG_20261023172343/20261023172343.bdf`。
- 单位转换状态：{'pass': 8}；原始 physical dimension 集合：['["uV"]']。
- MNE 返回的校准数据单位按库约定记录为 `V`；后续转换到 µV 使用每通道 header 单位对应的 multiplier。
- reference 可确认：0/8；montage 可确认：0/8。两者均写为 `unknown`，未根据通道名称推断。
- header 总 signal 数包含 BDF annotation signal；硬性 64 通道检查针对 MNE 暴露的 64 个数据通道。
- 每个文件的 record 数、每 record 样本数、record duration、物理/数字上下限及预期文件大小均已与实际读取结果核对。

## 阻止进入第四阶段的问题

- 无；8 条 recording 均通过本阶段文件级硬性检查。

## 已记录但推迟到第四阶段的信号级问题

- `SUB5` `SUB5_M/BGI_EEG_20261026161903/20261026161903.bdf`：["FC1"]
- `SUB8` `SUB8_22_M/BGI_EEG_20261029160859/20261029160859.bdf`：["FT7"]

8 条 recording 的大绝对幅值也作为 warning 记录，但未升级为本阶段文件读取失败，且没有在本阶段进行滤波、置零、插值或其他信号修改。

## Header 人口学字段

BDF header 的 `his_id=X`、sex unknown、birthday 等于采集日期均作为匿名占位/冲突 warning 记录。目录名仍是 subject_id、age、sex 的权威来源。详见 `metadata_conflicts.csv`。

## 问题统计

- error：0
- warning：58
- info：0

## 实际执行命令

```powershell
D:\miniconda\python.exe D:\repo\华大\scripts\validate_huada_raw_bdf.py --data-dir D:\repo\华大\data\resting_state --config D:\repo\华大\configs\huada_resting_preprocessing.toml --output-dir D:\repo\华大\preprocessing_audit\20261030_105520 --chunk-seconds 30
```

## 结论

8 名被试均进入 manifest，8 个 BDF 均完成从首样本到末样本的分块读取。文件格式、500 Hz、64 数据通道、通道名/顺序、header 维度、有限值和 µV 单位转换均通过硬性检查。reference 与 montage 不能从可靠来源确认，保持 `unknown`。已知常数通道和极端幅值留待第四阶段信号级 QC。

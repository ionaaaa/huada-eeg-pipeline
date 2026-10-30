$ErrorActionPreference = "Stop"
# LEGACY ONLY: historical single subject split. Recommended protocol is
# run_50m_sex_cv4_full.sh (sex_stratified_subject_cv4).
$InputLance = "<REPLACE_WITH_TENCENT_CLOUD_MODEL_TOKEN_LANCE_URI>"
$StorageOptions = "<REPLACE_WITH_TENCENT_CLOUD_STORAGE_OPTIONS_JSON>"
$Checkpoint = "<REPLACE_WITH_50M_PRETRAINED_CHECKPOINT>"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path "D:\repo\华大\downstream_runs" "50m_sex_full_$Stamp"

& "D:\miniconda\python.exe" "D:\repo\华大\scripts\downstream_50m\largeScale_fineTune.py" `
  --input-lance $InputLance `
  --storage-options $StorageOptions `
  --task sex_classification `
  --protocol legacy_single_split `
  --subject-split-manifest "D:\repo\华大\manifests\sex_subject_splits.csv" `
  --output-dir $OutputDir `
  --checkpoint $Checkpoint `
  --mode linear_probe `
  --epochs 5 `
  --batch-size 64 `
  --seed 42 `
  --split-seed 42

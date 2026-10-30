$ErrorActionPreference = "Stop"
# LEGACY ONLY: evaluates a historical single-split checkpoint.
$InputLance = "<REPLACE_WITH_TENCENT_CLOUD_MODEL_TOKEN_LANCE_URI>"
$StorageOptions = "<REPLACE_WITH_TENCENT_CLOUD_STORAGE_OPTIONS_JSON>"
$DownstreamCheckpoint = "<REPLACE_WITH_TRAINED_50M_SEX_CHECKPOINT>"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path "D:\repo\华大\downstream_evaluation" "50m_sex_$Stamp"

& "D:\miniconda\python.exe" "D:\repo\华大\scripts\downstream_50m\largeScale_fineTune.py" `
  --input-lance $InputLance `
  --storage-options $StorageOptions `
  --task sex_classification `
  --protocol legacy_single_split `
  --subject-split-manifest "D:\repo\华大\manifests\sex_subject_splits.csv" `
  --output-dir $OutputDir `
  --downstream-checkpoint $DownstreamCheckpoint `
  --mode linear_probe `
  --evaluate

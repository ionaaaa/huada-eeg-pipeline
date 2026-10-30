$ErrorActionPreference = "Stop"
$InputLance = "<REPLACE_WITH_TENCENT_CLOUD_MODEL_TOKEN_LANCE_URI>"
$StorageOptions = "<REPLACE_WITH_TENCENT_CLOUD_STORAGE_OPTIONS_JSON>"
$DownstreamCheckpoint = "<REPLACE_WITH_TRAINED_50M_AGE_CHECKPOINT>"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path "D:\repo\华大\downstream_evaluation" "50m_age_$Stamp"

& "D:\miniconda\python.exe" "D:\repo\华大\scripts\downstream_50m\largeScale_fineTune.py" `
  --input-lance $InputLance `
  --storage-options $StorageOptions `
  --task age_regression `
  --subject-split-manifest "D:\repo\华大\manifests\age_subject_splits.csv" `
  --output-dir $OutputDir `
  --downstream-checkpoint $DownstreamCheckpoint `
  --mode linear_probe `
  --evaluate

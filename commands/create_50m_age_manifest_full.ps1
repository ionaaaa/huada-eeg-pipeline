$ErrorActionPreference = "Stop"
$InputLance = "<REPLACE_WITH_TENCENT_CLOUD_MODEL_TOKEN_LANCE_URI>"
$StorageOptions = "<REPLACE_WITH_TENCENT_CLOUD_STORAGE_OPTIONS_JSON>"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ManifestDir = Join-Path "D:\repo\华大\manifests" "formal_age_$Stamp"

& "D:\miniconda\python.exe" "D:\repo\华大\scripts\create_downstream_manifest.py" `
  --input-lance $InputLance `
  --storage-options $StorageOptions `
  --task age_regression `
  --seed 42 `
  --output-manifest (Join-Path $ManifestDir "age_subject_manifest.csv") `
  --output-splits (Join-Path $ManifestDir "age_subject_splits.csv")

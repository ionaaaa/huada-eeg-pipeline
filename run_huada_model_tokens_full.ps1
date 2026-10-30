$ErrorActionPreference = "Stop"

& "D:\miniconda\python.exe" "D:\repo\华大\scripts\build_huada_model_tokens.py" `
  --input-lance-path "D:\repo\华大\preprocessed_common\20261030_113220\common_windows.lance" `
  --output-lance-path "D:\repo\华大\model_tokens_full\20261030_134500\huada_model_tokens.lance" `
  --reference-50m-repo "D:\repo\xxxxxxxx" `
  --reference-1b-repo "D:\repo\EEG_pretrain"

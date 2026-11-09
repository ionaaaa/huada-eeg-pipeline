# Huada native LaBraM/REVE backends

## Boundary

These backends start from Huada BDF data read by MNE in volts. They never read
the existing FBMAE token Lance. That Lance represents a different contract:
10-second windows, `clip_divide_200`, and FBMAE tokenization.

The audited Huada BDF channel axis is fixed at 64 names in
`configs/huada_labram_target_input_contract_v1.json`. Its order differs from
the Dortmund source axis. A count of 64 is not compatibility evidence.

## LaBraM chain

1. Require a regular BDF with the exact audited Huada channel order and 500 Hz.
2. Read MNE volts and require finite `[64,T]` floating data.
3. Apply 0.1–75 Hz zero-phase FIR bandpass.
4. Apply the 50 Hz harmonics that are below both the bandpass high edge and
   source Nyquist. For v1 this is exactly `[50]`.
5. Polyphase resample 500→200 Hz.
6. Multiply once by 10000, before the full-recording cache.
7. Cache one complete float32 recording.
8. Slice non-overlapping 400-sample (2-second) windows.
9. Map the exact Huada names to LaBraM names. Position tensors are validity
   markers only (`[0,0,0]` valid, `-0.1` invalid); they do not claim a physical
   Huada montage.

The two `channel_wrapper.*` buffers in the portable checkpoint describe the
Dortmund source adapter. The Huada backend verifies their presence but never
loads them as the target adapter. It rebuilds the Huada mask and name mapping.

## REVE chain

1. Require the same exact BDF/channel/rate conditions.
2. Apply 0.5–99.5 Hz zero-phase FIR bandpass; no notch is allowed.
3. Polyphase resample 500→200 Hz.
4. Fit StandardScaler statistics independently for every channel using the
   entire resampled recording (`ddof=0`). Constant channels use scale 1 and
   therefore become zero, matching StandardScaler semantics.
5. Clip to `[-15,15]`; amplitude scale remains 1.
6. Cache one complete float32 standardized/clipped recording.
7. Slice non-overlapping 400-sample windows.
8. Inject a Huada-ordered static `[64,3]` position bank on every forward.
   `position_mask` is null.

The Dortmund position bank is never reused. The released REVE positions asset
is selected by Huada channel name into Huada order and fingerprinted over the
channel names plus little-endian float32 bytes.

## REVE release gate

The verified REVE `positions.json` asset is not stored in this local repo. Its
required SHA256 is
`f62ca8d7dd4b88e1422ea17348440a41ed0b73b9ff7d12c3b1229492c9b448a5`.
Consequently the repository contains a blocker record, not a fabricated
formal contract:
`configs/huada_reve_target_input_contract_v1.release_blocker.json`.

On a host with the verified asset, freeze the real target contract with:

```bash
python -m scripts.native_backends.freeze_huada_target_contract \
  --model-family reve \
  --reve-position-source /absolute/path/to/positions.json \
  --expected-reve-position-source-sha256 f62ca8d7dd4b88e1422ea17348440a41ed0b73b9ff7d12c3b1229492c9b448a5 \
  --output configs/huada_reve_target_input_contract_v1.json
```

The command does not open a BDF or model. It refuses an incomplete SHA, a
missing channel, duplicate target mapping, non-finite coordinate, wrong shape,
or an existing output. Until that released contract exists, formal REVE
backend construction fails closed.

## Full-recording cache

Each cache is an atomic directory containing:

- `recording.float32.npy` (pickle disabled, mmap-compatible);
- `metadata.json` (family, full scaler state, processing order, hashes);
- `_SUCCESS` written only after safe read-back.

The cache writer refuses overwrite. It publishes by renaming a sibling
temporary directory and removes the temporary directory on failure.

Example dry run (does not open BDF data):

```bash
python -m scripts.native_backends.build_huada_native_recording_cache \
  --model-family labram \
  --source-bdf /path/to/SUB1.bdf \
  --recording-id SUB1 \
  --target-input-contract configs/huada_labram_target_input_contract_v1.json \
  --expected-target-contract-file-sha256 <64-hex-file-sha256> \
  --output-dir /new/cache/root/SUB1 \
  --dry-run
```

Without `--dry-run`, only the named BDF is read. The destination parent must
already exist and the destination itself must not exist.

## Portable runtime

`build_native_age_backend()` accepts an already validated
`PortableAgeCheckpoint` and a released Huada target contract. It constructs
the family architecture without Lightning, strictly loads every model tensor
and the exact `weight`/`bias` age probe, applies mean pooling, and restores age
using the full-precision scaler in the portable checkpoint.

Expected outputs are LaBraM `[B,129,200]` and REVE `[B,128,512]`. Backend input
is strictly finite float32 `[B,64,400]`; a 10-second FBMAE window is rejected.

## Strict Dortmund-age zero-shot

`scripts.native_backends.run_huada_portable_age_zero_shot` consumes only a
completed native full-recording cache, its exact frozen Huada target contract,
a formal portable checkpoint plus both sidecars, and a passed NeuralBench
dynamic-parity directory. It refuses fixture checkpoints, incomplete caches,
SHA drift, family drift, source-checkpoint drift, or parity that is not passed.

Every recording is sliced into non-overlapping float32 `[64,400]` windows.
Only the window tensor enters the backend. The fixed Huada subject ages are
attached after forward; SUB5 remains null and is retained in prediction output
but excluded from every metric. The primary result is the mean of window-level
age predictions per subject. Reported metrics are MAE, RMSE, R-squared,
Pearson, Spearman, and signed bias over the seven labeled subjects. The fixed
27.5-year Dortmund source-mean baseline is always emitted.

The entry point contains no optimizer, scheduler, backward, calibration, or
prediction clipping path. It independently reconstructs each age as
`predicted_z * checkpoint_std + checkpoint_mean`, checks the backend result,
and fingerprints the model and native cache before and after inference.

Cloud wrappers are:

- `commands/run_huada_labram_dortmund_age_zero_shot.sh`;
- `commands/run_huada_reve_dortmund_age_zero_shot.sh`.

Both require the already-passed FBMAE strict zero-shot result. The REVE wrapper
also requires the completed LaBraM result, so its output contains the final
FBMAE/LaBraM/REVE subject-mean comparison without rerunning either reference.

# Portable age full checkpoint v1

## Purpose and boundary

`portable_age_full_checkpoint` is the cross-repository exchange format for:

```text
NeuralBench source full fine-tune
  -> exporter-owned canonical remap
  -> portable checkpoint
  -> Huada zero-shot / feature cache / probe transfer
```

It is not a Lightning resume file, optimizer or scheduler checkpoint, head-only
checkpoint, feature cache, RNG snapshot, or complete training-log archive.  It
contains only tensors and basic Python values and must be readable with
`torch.load(..., map_location="cpu", weights_only=True)`.  No consumer needs a
Lightning class or a NeuralBench wrapper prefix.

The fixed identity is:

| Field | v1 value | Required |
| --- | --- | --- |
| `schema_name` | `portable_age_full_checkpoint` | yes |
| `schema_version` | `1` | yes |
| `checkpoint_kind` | `portable_age_full_finetuned` | yes |
| `validation_mode` | `formal` or explicitly `fixture` | optional; defaults formal |

Formal loading is the default. `fixture` is accepted only when the caller sets
`allow_fixture=True`; it can never be mistaken for a complete model.

## Complete field structure

| Top-level field | Required content |
| --- | --- |
| `task` | exact age regression contract: `name`, `output_dim=1`, `target_name=age`, `target_unit=year` |
| `model` | `model_name`, `model_family`, JSON-only `model_config` sufficient to reconstruct the backbone |
| `backbone_state_dict` | non-empty canonical model keys to tensors |
| `age_probe_state_dict` | exactly `weight` `[1,feature_dim]` and `bias` `[1]` |
| `input_contract` | version, input kind/rate/window/dtype/shape/channel semantics, electrode vocabulary identity, canonicalization identity, mask identity, preprocessing identity |
| `representation_contract` | version, representation name/layer, pooling, `feature_dim`, float32 representation dtype, mask usage |
| `target_transform` | `type=zscore`, full-precision mean/std, `ddof=0`, equal-subject weighting, unit year |
| `source_training` | dataset/fingerprint/split SHA, positive train/validation/test subject counts, nonnegative best epoch, validation selection metric, JSON metrics |
| `source_checkpoint` | source format, source file SHA-256, positive source size |
| `provenance` | exporter/repository/git/code/time/Python/PyTorch identity |
| `integrity` | two state fingerprints and four canonical metadata hashes |

All metadata may contain only `str`, `int`, finite `float`, `bool`, `None`,
`list`, and string-keyed `dict`. `Path`, NumPy, Enum, dataclass, OmegaConf,
Lightning, and custom objects are forbidden. Undefined metrics use an object
such as `{"value":null,"defined":false,"reason":"..."}`; NaN and infinity are
never serialized.

### Nested field catalogue

Every field below is required in v1 unless marked optional.

| Object | Fields |
| --- | --- |
| `task` | `name`, `output_dim`, `target_name`, `target_unit` |
| `model` | `model_name`, `model_family`, `model_config` |
| `input_contract` | `contract_version=1`, `input_kind`, `sampling_rate_hz`, `window_seconds`, `input_dtype`, `shape_semantics`, `channel_semantics`, `electrode_vocabulary_name`, `electrode_vocabulary_sha256`, `channel_canonicalization_version`, `channel_canonicalization_sha256`, `mask_semantics_version`, `mask_semantics_sha256`, `preprocessing_profile`, `preprocessing_config_sha256` |
| `representation_contract` | `contract_version=1`, `representation_name`, `representation_layer`, `pooling`, `feature_dim`, `representation_dtype=float32`, `uses_token_valid_mask` |
| `target_transform` | `type`, `mean`, `std`, `ddof`, `equal_subject_weight`, `target_unit` |
| `source_training` | `dataset_name`, `dataset_fingerprint`, `split_manifest_sha256`, `train_subject_count`, `validation_subject_count`, `test_subject_count`, `best_epoch`, `selection_metric`, `validation_metrics`, `test_metrics` |
| `source_checkpoint` | `checkpoint_format`, `source_checkpoint_size_bytes`; `source_checkpoint_sha256` is formal-required and fixture-optional |
| `provenance` | `exporter_name`, `exporter_version`, `source_repository`, `source_git_dirty`, `created_at_utc`, `python_version`, `pytorch_version`; `source_git_commit` and `critical_code_sha256` are formal-required and fixture-optional |
| `integrity` | `backbone_state_fingerprint`, `age_probe_state_fingerprint`, `model_config_sha256`, `input_contract_sha256`, `representation_contract_sha256`, `target_transform_sha256` |

Family validators may require additional JSON fields inside `model_config`,
`input_contract`, or `representation_contract`; they may not reinterpret the
common fields.

## Canonical state keys

The exporter must remove all training-framework wrappers before saving. Keys
starting with `module.`, `model.`, `model.wrapped_model.`, or
`_lightning_module.` are rejected. The Huada loader does not guess prefixes,
suffix-match, or partially load a state.

For current FBMAE, `backbone_state_dict` is the direct
`EEGPretrainModel.state_dict()`: canonical roots are `tokenizer.`,
`channel_embed.`, `time_embed.`, `encoder.encoder.layers.`, and `head.head.`.
The probe is separately represented as `weight` and `bias`. NeuralBench's
existing `model.wrapped_model.backbone.*` and exact `model.probe.*` names must
therefore be remapped by the exporter, not the consumer.

For NeuralBench FFT exports, mapping is likewise exact and family-specific:

| Source key | Portable key |
| --- | --- |
| `model.probe.weight` / `.bias` | `age_probe_state_dict.weight` / `.bias` |
| `model.wrapped_model.model.<key>` | `backbone_state_dict.<key>` |
| LaBraM `model.wrapped_model._positionless_mask` | `channel_wrapper._positionless_mask` |
| LaBraM `model.wrapped_model._known_mask` | `channel_wrapper._known_mask` |
| REVE `model.wrapped_model.channel_indices` (only if materialized) | `channel_wrapper.channel_indices` |

Suffix matching, shape-only head discovery, and partial `strict=False` loading
are forbidden. REVE `final_layer.1` and `final_layer.2` remain in the canonical
backbone. Metadata marks them frozen, outside the age forward path, excluded
from the age representation, and required for exact source-model reconstruction.

## Source scaler precision

`target_transform.mean` and `.std` are the source training-subject statistics,
with equal subject weight and population standard deviation (`ddof=0`). They
are stored at the full value actually used by training. For the LaBraM/REVE FFT
checkpoints this is the checkpoint tensor value `4.218591213226318` (float32),
not the report-rounded `4.218591`. The independently recomputed float64 value
`4.218591189948011` is recorded separately as `recomputed_std`; inference uses
`target_transform.std`, never the recomputed or rounded value.

## Input and representation contracts

The common schema does not make FBMAE's layer number, token count, electrode
IDs, or masked-mean pooling universal. Those details live in
`model.model_config`, `input_contract`, and `representation_contract` and are
checked by a registered family validator.

The current FBMAE validator requires the audited 50M construction
(`EEGPretrainModel`, 512 dimension, 12 layers, 8 heads, 64 channels, 10 time
patches, time-frequency objective), representation layer 8, masked mean,
float32 output, and `token_valid_mask_true_is_valid_v1`. A formal FBMAE state
must have all five key roots and the audited 154 tensors / 38,194,713
parameters. A fixture may use a small explicitly marked state but still checks
canonical roots and the 512-dimensional probe.

An FBMAE producer therefore emits fragments like:

```json
{
  "model": {
    "model_name": "fbmae_50m",
    "model_family": "fbmae",
    "model_config": {
      "class": "EEGPretrainModel", "input_dim": 100, "d_model": 512,
      "n_heads": 8, "depth": 12, "mlp_ratio": 4.0, "dropout": 0.1,
      "n_channels": 64, "n_time_patches": 10, "n_bands": 5,
      "frames_per_patch": 5, "objective": "timefreq"
    }
  },
  "representation_contract": {
    "contract_version": 1,
    "representation_name": "fbmae_encoder_token_embedding",
    "representation_layer": 8,
    "pooling": "masked_mean",
    "feature_dim": 512,
    "representation_dtype": "float32",
    "uses_token_valid_mask": true
  }
}
```

## Stable fingerprint algorithms

Tensor state fingerprint is SHA-256. Iterate keys in Unicode sort order. For
each tensor, append without separators: key UTF-8 bytes, canonical PyTorch dtype
string ASCII bytes (for example `torch.float32`), Python tuple shape ASCII
bytes, then bytes of `tensor.detach().cpu().contiguous()` viewed as `uint8`.
The operation does not mutate the source tensor.

Metadata/contract SHA-256 hashes UTF-8 JSON produced with
`ensure_ascii=False`, `sort_keys=True`, `separators=(",", ":")`, and
`allow_nan=False`. Hashes therefore do not depend on mapping insertion order,
the locale, or default platform encoding.

## Sidecar manifest

The portable file's own hash is intentionally not embedded, avoiding
self-reference. `<checkpoint>.manifest.json` is published last and contains
exactly:

```json
{
  "portable_file_sha256": "...",
  "portable_file_size_bytes": 123,
  "schema_name": "portable_age_full_checkpoint",
  "schema_version": 1,
  "checkpoint_kind": "portable_age_full_finetuned",
  "model_name": "fbmae_50m",
  "backbone_state_fingerprint": "...",
  "age_probe_state_fingerprint": "..."
}
```

The default loader requires the sidecar and recomputes every entry. `save`
validates before serialization, writes a unique temporary file, safe-loads and
fully validates it, stages the sidecar, rechecks the file hash, atomically
renames the checkpoint, and publishes the sidecar last. It rejects overwrite
unless explicitly enabled and never creates the destination directory.

## Producer and consumer responsibilities

The NeuralBench exporter owns source SHA verification, exact source-key audit,
canonical remap, extraction of full-precision scaler and source metadata,
JSON-type normalization, integrity calculation, safe save, and sidecar
publication. It must not export optimizer/scheduler/callback/loop/RNG state.

The Huada loader owns file/SHA checks before deserialization, weights-only CPU
load, complete common validation, hash/fingerprint recomputation, probe and
cross-field checks, family validation, and sidecar verification. It returns a
validated read-only view and never constructs a model, optimizer, scheduler,
or training framework object.

## Existing FBMAE full-checkpoint mapping

| Dortmund full checkpoint | Portable v1 |
| --- | --- |
| `backbone_state_dict` | `backbone_state_dict` (already canonical) |
| `age_probe_state_dict` | `age_probe_state_dict` |
| `model_architecture_config` | `model.model_config` plus `representation_contract` |
| `electrode_vocabulary` | vocabulary name/SHA in `input_contract` |
| `channel_canonicalization` | version/SHA in `input_contract` |
| `token_mask_semantics_version` | mask version/SHA in `input_contract` |
| `target_transform.source_age_mean/std` | `target_transform.mean/std` at full precision |
| `split_manifest_sha256`, source fingerprint | `source_training` |
| `best_epoch`, validation metrics | `source_training` |
| `git`, `critical_code_sha256` | `provenance` |
| source file SHA/size (external inventory) | `source_checkpoint` |
| optimizer/scheduler/gradient audit | deliberately excluded |

## LaBraM and REVE FFT contracts

Both validators are now registered. They require the released source-contract
SHA, 200 Hz, two-second `[B,64,400]` float32 input, the audited pretrained
snapshot revision/SHA, exact probe width, split semantic SHA, scaler precision,
finite tensors, canonical roots, and state fingerprint.

LaBraM requires cache payload `MNE volts ×10000`, extractor scaling exactly
once before cache, wrapper scale 1, `[B,129,200]` output, and an unmasked mean
over all 129 final tokens including CLS. Its two 64-element boolean channel
wrapper buffers are part of canonical portable state. Formal completeness is
223 backbone tensors and 5,817,336 non-wrapper model parameters. The source
downstream state has 225 tensors total: 223 backbone tensors plus the age
probe weight and bias. The two 64-element channel-wrapper masks are buffers
and are included in the 223 backbone tensors but not in the parameter count.

REVE records `amplitude_scale_factor = 1.0`; it intentionally does not claim
an extractor multiplier because the released source contract uses
`scale_factor = null` and caches full recordings after per-channel
StandardScaler normalization and clipping. Any `extractor_scale = 10000` or
LaBraM `volts_times_10000` cache semantics remain a hard validation failure.

REVE requires bandpass 0.5–99.5 Hz, no notch, per-recording/per-channel
`StandardScaler` statistics from the entire recording, clip `[-15,15]`, scale
1, null position mask, audited `[64,3]` position bank fingerprint, patch
length/overlap/step 200/20/180, `[B,128,512]` output, and an unmasked mean over
all 128 encoder tokens. Any LaBraM ×10000 semantics fail closed. Formal
completeness is 143 tensors and 69,451,266 parameters, including the frozen
source `final_layer.1/2` described above.

The shared machine-readable spec is
`schemas/portable_age_checkpoint_v1_spec.json`; its canonical JSON SHA is
`b7a77fcb5b359bb9621d9a800f7090fedef1f247a926d4d93ece8764c03c076f`.
The cross-repository drift sentinel uses a fixed tiny tensor state with
fingerprint `d0b559872ff8bd1258da9bec52ce53cd0dfc1b0c2fd99f447fb6f159f3a48325`.

## Versioning

Additive optional metadata that preserves all v1 meaning may remain v1 only
after loader support is defined. Removing or renaming required fields, changing
hash canonicalization, state-key meaning, tensor layout, target-transform
semantics, or validation behavior is incompatible and must increment
`schema_version`. A v1 loader rejects every other version.

## Fixture demonstration

This command only creates synthetic tensors in the named directory, writes the
sidecar, safe-loads it, and prints a summary:

```bash
python -m scripts.checkpoints.portable_age_checkpoint_demo --output-dir <temporary-fixture-dir>
```

It does not access a formal checkpoint, CFS, Lance data, a real model, forward,
zero-shot, feature cache, or CV7.

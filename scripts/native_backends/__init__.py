"""Native Huada signal pipelines and portable LaBraM/REVE runtimes."""

from .contracts import (
    HUADA_CHANNEL_AXIS,
    NativeContractError,
    load_huada_target_contract,
    position_bank_fingerprint,
    validate_huada_target_contract,
)
from .preprocessing import (
    NativeRecordingCache,
    iter_native_windows,
    preprocess_native_recording,
)
from .portable_age_backend import (
    HuadaLaBraMAgeBackend,
    HuadaREVEAgeBackend,
    build_native_age_backend,
)

__all__ = [
    "HUADA_CHANNEL_AXIS",
    "HuadaLaBraMAgeBackend",
    "HuadaREVEAgeBackend",
    "NativeContractError",
    "NativeRecordingCache",
    "build_native_age_backend",
    "iter_native_windows",
    "load_huada_target_contract",
    "position_bank_fingerprint",
    "preprocess_native_recording",
    "validate_huada_target_contract",
]

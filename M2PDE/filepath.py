import os
from pathlib import Path
from typing import Optional


def resolve_dataset_root(*, must_exist: bool = False) -> Path:
    """
    Resolve the root directory that stores the NTcouple datasets.

    Preference order:
    1. Environment variable ``NTCOUPLE_DATA_ROOT``.
    2. Repository-relative fallback: ``<repo>/M2PDE/cache_data/NTcouple``.

    Args:
        must_exist: When True, raise a FileNotFoundError if the resolved path does not exist.

    Returns:
        Path to the dataset root (not guaranteed to exist unless ``must_exist`` is True).

    Raises:
        FileNotFoundError: If ``must_exist`` is True and the path does not exist.
    """
    env_path: Optional[str] = os.environ.get("NTCOUPLE_DATA_ROOT")
    if env_path:
        candidate = Path(env_path).expanduser()
    else:
        candidate = Path(__file__).resolve().parents[1] / "cache_data" / "NTcouple"

    if must_exist and not candidate.exists():
        raise FileNotFoundError(
            f"NTcouple dataset root '{candidate}' not found. "
            "Set NTCOUPLE_DATA_ROOT or place the dataset under M2PDE/cache_data/NTcouple."
        )
    return candidate


# Backwards-compatible constant used throughout legacy code paths.
ABSOLUTE_PATH = str(resolve_dataset_root(must_exist=False))

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from nmp_modeling.data import infer_input_type


@dataclass(frozen=True)
class LoadedData:
    """Loaded empirical data with optional metadata."""
    data: object
    input_type: str | None = None
    tr: float | None = None
    metadata: dict | None = None
    source: str | None = None


def load_data(path, input_type=None, key=None, tr=None):
    """Load empirical data from common array or matrix file formats."""
    path = Path(path)
    suffix = _full_suffix(path)

    if suffix == ".npy":
        data = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        data = _load_npz(path, key=key)
    elif suffix in {".csv", ".txt"}:
        data = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    elif suffix == ".tsv":
        data = np.loadtxt(path, delimiter="\t")
    elif suffix == ".mat":
        data = _load_mat(path, key=key)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return LoadedData(
        data=data,
        input_type=input_type or infer_input_type(data),
        tr=tr,
        metadata={},
        source=str(path),
    )


def _full_suffix(path):
    """Return the full suffix, preserving .dtseries.nii-like endings."""
    name = path.name.lower()
    if name.endswith(".dtseries.nii"):
        return ".dtseries.nii"
    return path.suffix.lower()


def _load_npz(path, key=None):
    """Load an array from an NPZ file."""
    archive = np.load(path, allow_pickle=False)
    if key is not None:
        return archive[key]
    keys = list(archive.files)
    if len(keys) == 0:
        raise ValueError("NPZ file contains no arrays.")
    if len(keys) > 1:
        raise ValueError(
            "NPZ files with multiple arrays require a key. "
            f"Available keys: {keys}"
        )
    return archive[keys[0]]


def _load_mat(path, key=None):
    """Load an array from a MATLAB .mat file."""
    from hdf5storage import loadmat
    mat = loadmat(path)
    if key is not None:
        return mat[key]
    candidates = {
        k: v
        for k, v in mat.items()
        if not k.startswith("__") and isinstance(v, np.ndarray)
    }
    if len(candidates) == 0:
        raise ValueError("MAT file contains no array variables.")
    if len(candidates) > 1:
        raise ValueError(
            "MAT files with multiple array variables require a key. "
            f"Available keys: {sorted(candidates)}"
        )
    return next(iter(candidates.values()))

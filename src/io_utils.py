# src/io_utils.py
from __future__ import annotations

import json
import os
import tempfile
from typing import Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not exist. Accepts either a directory
    path or a file path ending; if the path looks like a file (has a
    suffix), its parent directory is created instead.
    """
    target = path
    # If the path ends with a filename-like token containing a dot and exists not as a dir,
    # assume caller passed a file path and create its parent.
    if os.path.splitext(path)[1] and not path.endswith(os.sep):
        target = os.path.dirname(path) or "."
    if target:
        os.makedirs(target, exist_ok=True)


def _atomic_write_bytes(payload: bytes, out_path: str) -> None:
    """
    Write bytes atomically: write to a temporary file in the same directory,
    then rename to the destination. Ensures durability and resists partial writes.
    """
    ensure_dir(out_path)
    d = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _atomic_write_text(text: str, out_path: str) -> None:
    _atomic_write_bytes(text.encode("utf-8"), out_path)


def save_json(obj, out_path: str) -> None:
    """
    Serialize obj to JSON (pretty, sorted keys) and write atomically.
    """
    payload = json.dumps(obj, indent=2, sort_keys=False).encode("utf-8")
    _atomic_write_bytes(payload, out_path)


def save_text(text: str, out_path: str) -> None:
    """
    Write UTF-8 text atomically.
    """
    _atomic_write_text(text, out_path)


def load_corr_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a correlator npz with:
      - C: shape (T, N, N), SPD per time
      - ts: optional 1D integer array of length T; defaults to arange(T)
    Validates shapes and returns (C.astype(float), ts:int).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with np.load(path) as z:
        if "C" not in z:
            raise ValueError(f"{path}: missing array 'C' with shape (T,N,N)")
        C = z["C"]
        if C.ndim != 3 or C.shape[1] != C.shape[2]:
            raise ValueError(f"{path}: 'C' must have shape (T,N,N); got {C.shape}")
        if "ts" in z:
            ts = z["ts"]
            if ts.ndim != 1 or ts.shape[0] != C.shape[0]:
                raise ValueError(f"{path}: 'ts' must be 1D of length T={C.shape[0]}; got {ts.shape}")
            ts = ts.astype(int, copy=False)
        else:
            ts = np.arange(C.shape[0], dtype=int)
    return C.astype(float, copy=False), ts
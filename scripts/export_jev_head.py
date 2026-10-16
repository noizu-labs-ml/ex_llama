#!/usr/bin/env python3
"""Export Jev's four FP32 head tensors to the portable native JEVH0001 format."""
import argparse
import hashlib
import struct
from pathlib import Path


def export(source, destination):
    import numpy as np
    if source.suffix == ".pt":
        import torch
        loaded = torch.load(source, map_location="cpu", weights_only=True)
        arrays = {key: loaded[key].detach().cpu().numpy() for key in
                  ("mu", "sd", "linear.weight", "linear.bias")}
    elif source.suffix == ".npz":
        with np.load(source, allow_pickle=False) as loaded:
            arrays = {key: loaded[key if key in loaded else key.removeprefix("linear.")]
                      for key in ("mu", "sd", "linear.weight", "linear.bias")}
    else:
        raise ValueError("head source must be .pt or .npz")
    arrays = {key: np.asarray(value, dtype="<f4") for key, value in arrays.items()}
    weights = arrays["linear.weight"]
    if weights.ndim != 2 or weights.shape[0] != 256 or not 1 <= weights.shape[1] <= 65536:
        raise ValueError("expected linear.weight shape (256, hidden)")
    hidden = weights.shape[1]
    if arrays["mu"].shape not in ((hidden,), (1, hidden)) or arrays["sd"].shape not in ((hidden,), (1, hidden)):
        raise ValueError("mu and sd must have hidden dimensions")
    if arrays["linear.bias"].shape != (256,):
        raise ValueError("expected linear.bias shape (256,)")
    if not all(np.isfinite(value).all() for value in arrays.values()) or not (arrays["sd"] > 0).all():
        raise ValueError("head parameters must be finite with positive standard deviations")
    data = b"JEVH0001" + struct.pack("<II", hidden, 256)
    data += b"".join(arrays[key].tobytes(order="C") for key in
                     ("mu", "sd", "linear.weight", "linear.bias"))
    destination.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print("Exported decision head SHA256:", export(args.source, args.destination))

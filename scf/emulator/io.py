"""HDF5 round-trip for :class:`XCNet`.  SPEC_scf §6 P3, §8.10 R3 (artifact handoff to P4).

R3 commits the trained UHF XCNet artifact to
``examples/scf_xc_emulator_uhf_<basis>.h5`` so P4 (GRAIL integration) can
pick it up without retraining.  The file format is keyed off the XCNet
config attributes (``n_unique``, ``hidden``, ``depth``); for the UHF path
``n_unique`` carries the doubled-seam length ``2 * M*(M+1)/2`` and is
self-describing.  We use HDF5 rather than ``torch.save``/``state_dict``
because:

  1. The rest of the codebase's persistent ensemble artifacts
     (``examples/two_level_residual_ensemble.h5``,
     ``examples/three_level_residual_ensemble.h5``,
     proposed Tier R references) use HDF5 — keeping XCNet artifacts in the
     same format simplifies later cross-artifact tooling.
  2. The h5py format is language-agnostic and self-describing, making it
     easier to inspect the trained-emulator file from a notebook or a
     non-Python consumer.
  3. The float32 parameter buffers are small (~5–20 kB total for STO-3G /
     6-31G) so any storage-efficiency advantage of binary torch is
     negligible.

The file layout is flat:

  /  (root)
    attrs:
      n_unique : int — XCNet config
      hidden   : int
      depth    : int
      grail_xcnet_version : "1.0"  — schema marker for forward-compatibility
    datasets:
      net.<i>.weight : float32 array — Linear layer i weight matrix
      net.<i>.bias   : float32 array — Linear layer i bias vector

Layer indices come from ``XCNet.net`` (``nn.Sequential``) and are sparse
(0, 2, 4, …) because GELU activations sit between the linear blocks.  We
preserve the original ``state_dict`` key names exactly so the round-trip
is byte-identical.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from .network import XCNet

_SCHEMA_VERSION = "1.0"


def save_xc_emulator(net: XCNet, path: str | Path) -> None:
    """Persist a trained :class:`XCNet` to HDF5.

    Writes the architecture config (``n_unique``, ``hidden``, ``depth``)
    as root-group attributes and each parameter tensor as a top-level
    float32 dataset keyed by the ``state_dict`` parameter name.

    Parameters
    ----------
    net : XCNet
        Trained network.  Parameters are read in their stored dtype
        (float32 by construction; see ``XCNet`` docstring).
    path : str or Path
        Output file.  Overwritten if it exists.

    Notes
    -----
    The grail_xcnet_version attribute is included so future schema bumps
    can be detected at load time.  We do not currently validate it on
    load (only one version exists); a strict check can be added when
    the schema first changes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net.eval()

    with h5py.File(path, "w") as f:
        f.attrs["grail_xcnet_version"] = _SCHEMA_VERSION
        f.attrs["n_unique"] = int(net.n_unique)
        f.attrs["hidden"] = int(net.hidden)
        f.attrs["depth"] = int(net.depth)

        for name, tensor in net.state_dict().items():
            # Convert to numpy float32 for storage.  Sanitize the key by
            # leaving it untouched: h5py treats dots in dataset names
            # fine, and matching state_dict exactly lets us load_state_dict
            # straight back without any string surgery.
            arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            f.create_dataset(name, data=arr)


def load_xc_emulator(path: str | Path, *, dtype: torch.dtype = torch.float32) -> XCNet:
    """Reconstruct an :class:`XCNet` from an HDF5 file written by
    :func:`save_xc_emulator`.

    Reads the architecture config from root attributes, instantiates a
    fresh ``XCNet`` with that shape, then ``load_state_dict``-loads every
    top-level dataset into the corresponding parameter.  After this call
    the returned network is in ``eval()`` mode and produces outputs
    identical to the network that was saved (modulo float32 round-trip,
    which is a no-op because the buffer dtype is unchanged).

    Parameters
    ----------
    path : str or Path
        Input HDF5 file.
    dtype : torch.dtype, default torch.float32
        Evaluation precision.  Float64 promotes the frozen float32
        weights exactly, with no retraining or checkpoint modification.

    Returns
    -------
    net : XCNet
        Reconstructed network in eval mode.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If a required attribute (``n_unique``, ``hidden``, ``depth``) or
        parameter dataset is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path!s}")

    with h5py.File(path, "r") as f:
        # Required architecture attributes.  Missing keys raise KeyError,
        # which is the desired failure mode for a malformed file.
        n_unique = int(f.attrs["n_unique"])
        hidden = int(f.attrs["hidden"])
        depth = int(f.attrs["depth"])

        net = XCNet(n_unique=n_unique, hidden=hidden, depth=depth)
        expected_keys = set(net.state_dict().keys())

        # Read every dataset; build a state_dict-compatible mapping.
        loaded: dict[str, torch.Tensor] = {}
        for key in f.keys():
            if key not in expected_keys:
                # Unknown dataset (forward-compat reserve): skip silently.
                # A strict-mode load could promote this to an error, but
                # for v1.0 we tolerate spectators.
                continue
            arr = np.asarray(f[key][...], dtype=np.float32)
            loaded[key] = torch.from_numpy(arr)

        missing = expected_keys - loaded.keys()
        if missing:
            raise KeyError(
                f"HDF5 file {path!s} is missing XCNet parameters: {sorted(missing)!r}."
            )

    # strict=True catches shape mismatches; the only way to slip past it
    # is to save and load with consistent (n_unique, hidden, depth) — which
    # the attributes guarantee.
    net.load_state_dict(loaded, strict=True)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("XCNet evaluation dtype must be float32 or float64")
    net.to(dtype=dtype)
    net.eval()
    return net

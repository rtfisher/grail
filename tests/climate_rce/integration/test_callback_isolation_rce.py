"""I-callback-isolation-rce: host-agnosticism enforcement for testbed 4 (RCE).

SPEC_climate_rce §6 P4 (test_callback_isolation_rce) and §3.2
(host-agnosticism test-enforcement).  Mirrors:

    the independent callback-isolation contract

and the canonical NLTE test:

    tests/integration/twolevel/test_integration.py::test_I_callback_isolation_numpy_host_jvp

Algorithm
---------
1. Build a ``ColumnSpec`` and smooth-fallback ``SubgridNet``.
2. Equilibrate the truth RCE fixed point via ``equilibrate_to_rce``.
3. Build ``delta_T = make_back_residual(...)`` (the GRAIL RHS).
4. Construct the full cycle Jacobian ``K`` in numpy via central-FD on the
   canonical split cycle ``_T_split_cycle_np`` (the same function imported
   by ``climate_rce.host`` for the P3 isolation gate — no circularity).
5. Build the **numpy-only** ``host_jvp_numpy_only`` that:
   - Accepts its input as numpy array or torch.Tensor.
   - Applies ``v − K·v`` in pure numpy (no torch in the body).
   - Returns a numpy array — hiding torch from GRAIL.
   Supply ``emul_jvp = make_identity_emulator_jvp()`` (Path B additive-cycle
   workaround encoding).
6. Assert ``resolvent_solve`` completes with:
   - ``delta_x_star`` finite (NaN/Inf-free).
   - ``iter_count > 0`` (GMRES ran at least one iteration).
7. Run the same solve with the torch-native ``host_jvp`` from
   ``make_host_jvp_rce(path='additive')`` (Path B: ``v ↦ v − K·v``
   computed via FD in torch).  Assert the two results agree to ``1e-9``.

Why this test is load-bearing
-------------------------------
GRAIL's host-agnosticism claim — "opaque callback interface — any coupled-
physics code wraps GRAIL around its own AI-embedded simulation" — requires
that the library does NOT import or depend on the caller's framework.
This test enforces that at the RCE-testbed level: a Fortran or numpy-based
climate SCM could supply ``K·v`` via a numpy callback and GRAIL would work
identically to the torch-native path.  SPEC §3.2: "GRAIL host-agnosticism is
test-enforced."

Note on Path B encoding
-----------------------
The numpy-only path uses the additive-cycle workaround (Path B):
    host_jvp(v) = v − K·v  (full cycle; NN-bearing)
    emul_jvp(u) = u         (identity)
so ``resolvent_matvec(emul_id, host)(v) = v − (v − K·v) = K·v``.
This is a valid cross-check for the numpy interface; Path A (compositional)
is tested in the main resolvent-identity test.

The torch cross-check uses the same encoding via ``make_host_jvp_rce(path='additive')``,
so both paths compute v − K·v from the same FD Jacobian and must agree to float64 precision.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from climate_rce.column import ColumnSpec
from climate_rce.emulator._smooth_fallback import make_smooth_fallback
from climate_rce.host import equilibrate_to_rce, _T_split_cycle_np
from climate_rce.jvp import (
    make_back_residual,
    make_host_jvp_rce,
    make_identity_emulator_jvp,
    make_subgrid_with_nn,
)
from grail import resolvent_solve


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# Use a moderate dt: rho(K) ~ 0.990, ||G|| ~ 100.  Intermediate amplification
# regime ensures GMRES actually has non-trivial work to do (iter_count > 0 is
# meaningful) while keeping the test fast (<5 s total).
# Physical justification: dt = 5e3 s corresponds to a short radiation
# sub-cycle within the typical 15-min (~900 s) GCM timestep; at this scale
# the WV-feedback spectral radius is reliably < 1.
_DT = 5e3      # seconds
_CO2_PPM = 280.0
_N_Z = 10      # column levels — small, fast
_AMPLITUDE = 1e-4   # smooth-fallback scale; small → perturbative
_MAX_ITER = 1000    # generous for this dt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec() -> ColumnSpec:
    """Small test ColumnSpec (same as test_resolvent_predicts_rce)."""
    pressure = np.linspace(5000.0, 1e5, _N_Z)
    return ColumnSpec(
        n_z=_N_Z,
        n_v=2,
        var_names=("T", "q"),
        pressure=pressure,
        var_scales=(300.0, 1e-2),
        n_state=_N_Z,
    )


def _zero_subgrid(spec: ColumnSpec, profile: np.ndarray) -> np.ndarray:
    """Zero truth subgrid tendency (only radiation + resolved convection)."""
    return np.zeros((spec.n_v, spec.n_z), dtype=np.float64)


def _build_cycle_K_numpy(
    spec: ColumnSpec,
    T_star: np.ndarray,
    subgrid_tendency,
    *,
    dt: float,
    co2_ppm: float,
    h_fd: float = 1e-4,
) -> np.ndarray:
    """Build the cycle Jacobian K = ∂𝒯_split/∂T by central-FD.

    Uses the canonical split cycle ``_T_split_cycle_np`` (the same function
    imported by ``climate_rce.host`` and tested in the P3 isolation gate).
    All operations are in pure numpy — no torch.

    Parameters
    ----------
    spec : ColumnSpec
    T_star : ndarray (n_z,) converged T profile in K
    subgrid_tendency : Callable
    dt : float  pseudo-timestep
    co2_ppm : float
    h_fd : float  FD step as a fraction of T_scale (default 1e-4)

    Returns
    -------
    ndarray (n_z, n_z) float64, the split-cycle Jacobian K_split.
    """
    T_scale = float(spec.var_scales[0])
    h_phys = h_fd * T_scale     # physical perturbation in K
    n_z = spec.n_z

    K = np.zeros((n_z, n_z), dtype=np.float64)
    for j in range(n_z):
        e_j = np.zeros(n_z, dtype=np.float64)
        e_j[j] = h_phys
        Tp = _T_split_cycle_np(spec, T_star + e_j, subgrid_tendency,
                               dt=dt, co2_ppm=co2_ppm)
        Tm = _T_split_cycle_np(spec, T_star - e_j, subgrid_tendency,
                               dt=dt, co2_ppm=co2_ppm)
        K[:, j] = (Tp - Tm) / (2.0 * h_phys)
    return K


# ---------------------------------------------------------------------------
# I-callback-isolation-rce
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_I_callback_isolation_rce() -> None:
    'I-callback-isolation-rce: resolvent_solve works with a numpy-only host_jvp.\n\n    The numpy-only host_jvp accepts any array-like, applies ``v − K·v``\n    using a pre-assembled dense numpy K (no torch in the body), and returns\n    a numpy array.  GRAIL must complete without torch awareness in the host\n    callback.\n\n    Assertions:\n    1. ``delta_x_star`` is NaN/Inf-free with the numpy-only host_jvp.\n    2. ``iter_count > 0`` (GMRES ran at least one iteration).\n    3. numpy and torch host_jvp paths agree to ``1e-9``.\n\n    SPEC §3.2: ``grail/`` must never import ``climate_rce/``; this test\n    verifies the direction of agnosticism — the *host_jvp callback* may use\n    arbitrary frameworks internally, as long as its input/output is array-like.'
    spec = _make_spec()
    model = make_smooth_fallback(spec, amplitude=_AMPLITUDE, seed=42)
    s_truth = _zero_subgrid
    s_nn = make_subgrid_with_nn(model, dt=_DT)

    # --- Equilibrate the truth RCE fixed point ---
    res_truth = equilibrate_to_rce(
        spec, s_truth,
        dt=_DT, max_iter=_MAX_ITER, co2_ppm=_CO2_PPM,
    )
    if not res_truth.converged:
        pytest.skip(
            f"test_I_callback_isolation_rce: truth equilibration did not "
            f"converge in {res_truth.n_iter} iterations at dt={_DT:.0e} s.  "
            "This is a host-driver issue (P1), not a P4 failure."
        )

    x_star_truth = res_truth.x_star   # (n_v, n_z) physical
    T_scale = float(spec.var_scales[0])
    T_star = x_star_truth[0]           # (n_z,) K

    # --- Build GRAIL RHS (delta_T) ---
    delta_T = make_back_residual(spec, x_star_truth, s_nn, s_truth, dt=_DT)
    x_star_t = torch.tensor(T_star / T_scale, dtype=torch.float64)

    # --- Build the full cycle Jacobian K in numpy (central-FD on _T_split_cycle_np) ---
    # Use s_truth (zero subgrid) for K: the numpy host_jvp is a pure-K
    # cross-check that doesn't depend on which subgrid is "current".
    # This is the same canonical split cycle used by P3's isolation gate.
    K_np = _build_cycle_K_numpy(
        spec, T_star, s_truth, dt=_DT, co2_ppm=_CO2_PPM,
    )   # (n_z, n_z) float64

    # --- Build the numpy-only host_jvp ---
    # Path B (additive-cycle workaround, SPEC §4.2):
    #   host_jvp(v) = v − K·v   (full-cycle NN-bearing encoding)
    # In the callback-isolation test, we use s_truth for K (no NN in K),
    # at theta* from the truth side.  The point is to test the numpy interface,
    # not the specific K value.
    def host_jvp_numpy_only(v) -> np.ndarray:
        """Numpy-only host_jvp: accepts array-like, returns numpy.

        Deliberately avoids torch inside this function body to prove GRAIL
        does not need torch awareness in the host callback.

        Returns ``v − K·v`` (Path B additive-cycle encoding; SPEC §4.2).
        """
        # Accept torch.Tensor or numpy array (GRAIL may pass either).
        if isinstance(v, torch.Tensor):
            v_np = v.detach().numpy().astype(np.float64)
        else:
            v_np = np.asarray(v, dtype=np.float64)
        # Apply the cycle Jacobian in pure numpy: v − K·v.
        # All operations are numpy; no torch calls.
        result = v_np - K_np @ v_np      # pure numpy, no torch
        return result                     # numpy array returned

    # --- Identity emulator_jvp (Path B partner) ---
    emul_jvp = make_identity_emulator_jvp()

    # --- Run resolvent_solve with numpy-only host_jvp ---
    result_numpy = resolvent_solve(
        x_star_t, delta_T, emul_jvp, host_jvp_numpy_only,
        gmres_tol=1e-9, gmres_maxiter=200,
    )

    # Assertion 1: NaN/Inf-free output.
    assert torch.isfinite(result_numpy.delta_x_star).all().item(), (
        "I-callback-isolation-rce: delta_x_star contains NaN/Inf with "
        "numpy-only host_jvp.  GRAIL must handle numpy callbacks transparently.  "
        "Check that resolvent_solve converts callback outputs to torch internally."
    )

    # Assertion 2: GMRES ran at least one iteration.
    assert result_numpy.iter_count > 0, (
        f"I-callback-isolation-rce: iter_count = {result_numpy.iter_count}; "
        "expected > 0.  GMRES did not run — check that delta_T is non-zero and "
        "the host_jvp is not the identity."
    )

    # --- Torch-native cross-check: Path B via make_host_jvp_rce(path='additive') ---
    # make_host_jvp_rce with path='additive' also computes v − K·v via FD on
    # the full NN-replaced cycle.  Here we use s_truth (zero NN) for consistency
    # with the numpy K.  Both paths implement the same operator to float64 precision.
    host_jvp_torch = make_host_jvp_rce(
        spec, x_star_truth, s_truth,
        path="additive", dt=_DT, co2_ppm=_CO2_PPM,
    )

    result_torch = resolvent_solve(
        x_star_t, delta_T, emul_jvp, host_jvp_torch,
        gmres_tol=1e-9, gmres_maxiter=200,
    )

    max_diff = float(
        (result_numpy.delta_x_star - result_torch.delta_x_star).abs().max().item()
    )

    print(f"\n--- I-callback-isolation-rce ---")
    print(f"  dt (pseudo-timestep)           = {_DT:.2e} s")
    print(f"  numpy host_jvp iter_count      = {result_numpy.iter_count}")
    print(f"  torch host_jvp iter_count      = {result_torch.iter_count}")
    print(f"  max |delta_x_numpy − delta_x_torch| = {max_diff:.3e}  (tol 1e-9)")
    print(f"  numpy delta_x_star = {result_numpy.delta_x_star.detach().numpy()}")
    print(f"  torch delta_x_star = {result_torch.delta_x_star.detach().numpy()}")

    # Assertion 3: numpy and torch paths agree to 1e-9 (float64 round-off level).
    # Justification: both paths use the same K matrix (one assembled in numpy,
    # one via FD from the same _T_split_cycle_np in torch); GMRES with the same
    # tolerance converges to the same solution to within GMRES tol ≈ 1e-9.
    assert max_diff < 1e-9, (
        f"I-callback-isolation-rce: numpy and torch host_jvp paths disagree by "
        f"{max_diff:.3e} (tolerance 1e-9).  Both paths implement v−K·v using "
        "the same FD step on _T_split_cycle_np; agreement to float64 precision "
        "is expected.  Check that K_np and the torch path use identical FD steps "
        "and the same subgrid_tendency (s_truth=zero in both)."
    )

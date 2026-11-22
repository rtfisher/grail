"""Current passive NLTE spectrum measurement (historical filename retained).

The former ambient population h=1e-6/30-step Arnoldi estimator is retired:
its stencil left the physical tangent neighborhood and its Ritz estimates
were unconverged. Historical NPZ data remain untouched. The current default
uses the independently checked positive linear-SC source operator and a dense
eigensolve, with an explicit eigen residual and actual finite-slab boundary.
Scalar spectral gain is labeled separately from the inverse operator 2-norm.
Joint refinement is recorded in the dated remediation evidence; even a dense
eigensolve is only a discretized measurement, not a continuum benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, svdvals

from atmosphere.twolevel.diagnostics import validated_atmosphere, independent_transfer, truth_source


def measure_rho_K(eps: float, n_tau=640, n_nu=81) -> dict:
    """Measure the checked physical tangent in independent source coordinates."""
    atmosphere=validated_atmosphere(T0=8000.,ne0=1e12,epsilon=eps,n_tau=n_tau,n_nu=n_nu)
    L,b,*_=independent_transfer(atmosphere)
    K=(1-eps)*L
    eigenvalues,eigenvectors=eig(K)
    idx=np.argmax(np.abs(eigenvalues));value=eigenvalues[idx];vector=eigenvectors[:,idx]
    rho=float(abs(value))
    source=truth_source(atmosphere,(L,b))
    A=np.eye(n_tau)-K
    return {"eps":eps,"rho_K":rho,"rho_K_real":float(value.real),
        "eigen_residual":float(np.linalg.norm(K@vector-value*vector)/np.linalg.norm(vector)),
        "scalar_spectral_gain":float(1/(1-rho)),
        "source_inverse_2norm":float(1/svdvals(A)[-1]),
        "n_tau":n_tau,"n_nu":n_nu,"n_mu":3,"formal_mode":"linear",
        "coordinates":"independent S/B0; eigenvalues similar to conserving n2 chart at root",
        "source_over_B_min":float(source.min()),"source_over_B_max":float(source.max()),
        "maximum_affine_transport_row_sum":float(np.max(L.sum(1)+b)),
        "true_equation_relative_defect":float(np.linalg.norm(A@source-((1-eps)*b+eps))/np.linalg.norm((1-eps)*b+eps))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-tau",type=int,default=640)
    parser.add_argument("--n-nu",type=int,default=81)
    parser.add_argument("--output",type=Path,default=Path("examples/output/remediation/2026-11-16/nlte/current_rho_K_dense.npz"))
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("preserve existing evidence; supply a new --output path")
    rows=[measure_rho_K(eps,args.n_tau,args.n_nu) for eps in [1e-2,1e-3,1e-4,1e-5]]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,epsilon=[r["eps"] for r in rows],
        rho_K=[r["rho_K"] for r in rows],scalar_spectral_gain=[r["scalar_spectral_gain"] for r in rows],
        source_inverse_2norm=[r["source_inverse_2norm"] for r in rows],eigen_residual=[r["eigen_residual"] for r in rows])
    payload={"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"rows":rows}
    args.output.with_suffix(".json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))


if __name__=="__main__":main()

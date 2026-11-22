# Lean verification of fixed-point response

The seven Lean modules in `lean/` prove conditional results about smooth
fixed-point response, quadratic remainders, Jacobian substitution, Born-series
summation, amplification, and exchangeable split-conformal coverage.

The project pins Lean `4.36.0-rc2` and the exact Mathlib/dependency revisions in
`lake-manifest.json`. After installing Elan and the pinned toolchain:

```sh
cd formalization/grail
lake exe cache get
lake build
python3 tools/kernel_check.py
```

`kernel_check.py` compiles each local module sequentially, checks the pinned
Mathlib revision, and inspects the transitive axiom footprints of the listed
declarations. It rejects `sorryAx` and axioms beyond `propext`,
`Classical.choice`, and `Quot.sound`. Reports are generated under `.lake/`.
An existing matching dependency cache may be supplied with
`--dependency-root <directory-containing-mathlib-and-other-packages>`.

The compiler checks the formal statements and proof terms. These standalone
checks do not establish semantic correspondence to an external English text,
nor do they certify trained weights, physical models, floating-point programs,
or exchangeability of a particular dataset. They do not run Span or include
external review records.

The finite-dimensional matrix results and the Banach-space local-response
results keep their assumptions explicit in their Lean declarations. The list
of headline declarations and module build order is in `theorems.json`.

"""Back-compat shim: re-exports from atmosphere.twolevel.rate_equations.

The canonical implementation has moved to atmosphere/twolevel/rate_equations.py
as part of SPEC_threelevel §M3a.  All existing imports of the form
``from atmosphere.rate_equations import ...`` continue to work unchanged via
this re-export.

Private constants (_H, _K, _C) are also re-exported because test_rate_equations.py
imports them directly for use in analytic-limit checks.
"""

from .twolevel.rate_equations import (  # noqa: F401
    _C,
    _H,
    _K,
    rate_equation_solve,
)

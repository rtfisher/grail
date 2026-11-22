"""Controlled smooth subgrid perturbation for the climate reference model."""
from climate_rce.column import ColumnSpec
from .network import SubgridNet
from ._smooth_fallback import make_smooth_fallback

__all__ = ['SubgridNet', 'make_smooth_fallback', 'load_subgrid_emulator']


def load_subgrid_emulator(spec: ColumnSpec, *, fmt: str = 'auto') -> SubgridNet:
    """Construct the controlled smooth surrogate from its column specification."""
    if not isinstance(spec, ColumnSpec) or fmt not in {'auto', 'smooth_fallback'}:
        raise ValueError('Use a ColumnSpec and the controlled smooth surrogate')
    return make_smooth_fallback(spec)

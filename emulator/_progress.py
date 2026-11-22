"""Lightweight progress-bar helper shared across training loops and sweeps.

Wraps ``tqdm`` when it is installed *and* the caller opts in; otherwise it
degrades to a no-op pass-through.  Two consequences that make this safe to
sprinkle anywhere:

* importing this module never hard-fails (``tqdm`` is treated as optional);
* ``enable=False`` is byte-identical in behaviour to the un-instrumented
  loop, so the default-off call sites keep the test suite silent and leave
  existing stdout untouched.

Usage
-----
Iterator wrapper::

    for epoch in progress_bar(range(n_epochs), desc="train", enable=verbose):
        ...

Manual bar (when there is no single iterable to wrap)::

    bar = progress_bar(total=n_configs, desc="sweep", enable=verbose)
    for cfg in configs:
        ...
        bar.set_postfix(pass_rate=f"{k}/{n}")
        bar.update()
    bar.close()
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

try:  # tqdm is in the pinned conda env, but keep it optional.
    from tqdm.auto import tqdm as _tqdm
    _HAVE_TQDM = True
except Exception:  # pragma: no cover - exercised only on tqdm-less envs
    _HAVE_TQDM = False


class _NullBar:
    """Pass-through stand-in for tqdm when progress is disabled/unavailable.

    Iterating it yields the wrapped iterable unchanged; every tqdm control
    method (``set_postfix``, ``update``, ``close`` ...) is a no-op.  Supports
    use as a context manager so call sites need no ``if enable`` guards.
    """

    def __init__(self, iterable: Optional[Iterable] = None) -> None:
        self._it = iterable if iterable is not None else ()

    def __iter__(self):
        return iter(self._it)

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    def write(self, *args: Any, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "_NullBar":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


def progress_bar(
    iterable: Optional[Iterable] = None,
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    enable: bool = True,
    **kwargs: Any,
):
    """Return a tqdm bar when enabled and available, else a no-op bar.

    Parameters
    ----------
    iterable : iterable, optional
        Sequence to wrap.  Omit for a manual bar driven by ``update()``.
    total : int, optional
        Item count for a manual bar (or when ``len(iterable)`` is unknown).
    desc : str, optional
        Short label shown to the left of the bar.
    enable : bool, default True
        When False (or when tqdm is unavailable) a :class:`_NullBar` is
        returned, making the call site behave exactly as if uninstrumented.
    **kwargs
        Forwarded verbatim to ``tqdm`` (e.g. ``unit``, ``leave``).
    """
    if enable and _HAVE_TQDM:
        return _tqdm(iterable, total=total, desc=desc, **kwargs)
    return _NullBar(iterable)

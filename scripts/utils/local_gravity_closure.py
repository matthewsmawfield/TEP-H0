"""Retired local-gravity mapping.

No Solar-System source-charge or PPN mapping is derivable from the
phenomenological TEP-H0 observable coefficient without a specified microscopic
action and solved source profiles. The former DGP-style Vainshtein calculation
has been removed from the authoritative analysis.
"""

from __future__ import annotations


_EXCLUSION = (
    "Local-gravity closure is excluded: TEP-H0 does not specify A(phi), B(phi), "
    "V(phi), or screened Solar/Earth field profiles."
)


def alpha_clock_from_kappa(*args, **kwargs):
    raise RuntimeError(_EXCLUSION)


def compute_local_gravity_closure(*args, **kwargs):
    raise RuntimeError(_EXCLUSION)


def closure_to_dict(*args, **kwargs):
    raise RuntimeError(_EXCLUSION)


__all__ = [
    "alpha_clock_from_kappa",
    "closure_to_dict",
    "compute_local_gravity_closure",
]

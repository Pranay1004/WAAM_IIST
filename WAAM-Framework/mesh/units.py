"""mesh/units.py -- one unit convention for the whole framework.

THE PROBLEM THIS SOLVES
-----------------------
Three different modules guessed length units three different ways:

  * mesh/intelligent_mesher.py :  val < 1.0  -> metres-to-mm (x1000)
  * core/fluid/openfoam_coupling: val > 1.0  -> mm-to-metres (/1000)
  * codegen/edp_builder._validate: per-key thresholds (lx > 5, hz_base > 1, ...)

so the SAME number meant different things depending on who read it: 0.9 was
900 mm in one place and 0.9 mm in another, and a legitimate 0.9 mm feature
was catastrophically rescaled by the first rule.

THE CONVENTION
--------------
Solver parameters are SI (metres). GUI presets and ad-hoc inputs are usually
millimetres. A value is therefore interpreted as millimetres -- and divided by
1000 -- only when it is LARGER than any legitimate SI value for that quantity:

    lx, ly_base : > 5.0 m      (a 5 m WAAM plate does not exist)
    hz_base     : > 1.0 m
    ty_dep      : > 1.0 m
    h_bead*     : > 0.5 m

Anything at or below the threshold passes through untouched, so genuine
sub-millimetre SI values (0.0009 m) are never rescaled. Conversions are
REPORTED through the optional `warn` callable -- silent rescaling is what made
the old bugs undetectable.
"""

from __future__ import annotations

from typing import Callable, Optional

# Per-quantity mm-detection thresholds [SI units]. A value strictly greater
# than its threshold is assumed to be millimetres.
MM_THRESHOLD_M = {
    "lx": 5.0,
    "ly_base": 5.0,
    "ly": 5.0,
    "hz_base": 1.0,
    "ty_dep": 1.0,
    "h_bead": 0.5,
    "h_bead_first": 0.5,
    "h_bead_subseq": 0.5,
    "wire_diameter": 0.05,   # 50 mm wire does not exist; 1.2 means 1.2 mm
    "default": 1.0,
}


def threshold_for(key: str) -> float:
    """Detection threshold for a quantity name (metres)."""
    return float(MM_THRESHOLD_M.get(key, MM_THRESHOLD_M["default"]))


def dim_to_m(value, key: str = "default",
             warn: Optional[Callable[[str], None]] = None) -> float:
    """Interpret `value` as SI metres, converting from mm when unambiguous.

    Values above the per-key threshold are divided by 1000 (and reported);
    anything else passes through. Raises ValueError on non-numeric input
    instead of guessing.
    """
    try:
        v = float(value)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"dimension {key!r}: {value!r} is not a number") from ex
    thr = threshold_for(key)
    if v > thr:
        if warn is not None:
            warn(f"{key}={v:g} exceeds the {thr:g} m plausibility bound; "
                 f"interpreted as millimetres -> {v / 1000.0:.6g} m")
        return v / 1000.0
    return v


def is_mm(value, key: str = "default") -> bool:
    """True when `value` would be interpreted as millimetres by dim_to_m."""
    try:
        return float(value) > threshold_for(key)
    except (TypeError, ValueError):
        return False

#!/usr/bin/env python3
"""OS-aware typography for the WAAM Tkinter GUI.

Centralises all font choices so every window renders with a native,
available typeface per operating system instead of hardcoding macOS-only
``Helvetica Neue`` / ``Menlo``.

Detection
---------
``platform.system()`` is normalised once at import:

- ``Darwin``  -> ``macos``
- ``Windows`` -> ``windows``
- ``Linux``   -> ``linux``
- anything else -> ``other`` (falls back to Linux stacks)

Per-OS stacks (first entry is the preferred Tk family)::

    macos:   UI: Helvetica Neue, Helvetica, SF Pro Text
             Mono: Menlo, SF Mono, Monaco
    windows: UI: Segoe UI, Tahoma
             Mono: Consolas, Cascadia Mono, Lucida Console
    linux:   UI: DejaVu Sans, Ubuntu, Liberation Sans, Noto Sans, FreeSans
             Mono: DejaVu Sans Mono, Ubuntu Mono, Liberation Mono, Noto Sans Mono

Why these picks
---------------
- ``Helvetica Neue`` / ``Menlo`` do not exist on stock Windows/Linux, so
  Tk silently falls back to Arial/Courier with different metrics and the
  fixed ``width=10`` entries clip. Native picks keep metrics stable.
- ``DejaVu Sans / Sans Mono`` ship with virtually every Linux distro and
  cover the scientific glyphs used here (Lx subscripts, ``chi``,
  ``rho``, ``epsilon``, ``Delta``, ``<=``, ``degC``, ``10^-6``).
- ``Segoe UI`` / ``Consolas`` are guaranteed on Windows 10/11.
- Sizes are nudged per OS because the same point size does not render the
  same x-height: DejaVu runs wide (title -2pt to still fit 1300px),
  Segoe UI runs tall (title -1pt).

Usage
-----
``gui/app.py`` (and any widget) should do::

    from gui.fonts import (
        FONT_TITLE, FONT_SUBTITLE, FONT_SECTION, FONT_LABEL, FONT_ENTRY,
        FONT_CONSOLE, FONT_STATUS, FONT_BTN_RUN, FONT_BTN, FONT_TAB,
        FONT_MONO_SM, FONT_SMALL, FONT_SMALL_BOLD, FONT_UNIT,
        FONT_MONO, FONT_INFO,
        OS_TYPE, UI_FONT_FAMILY, MONO_FONT_FAMILY,
        apply_os_typography,
    )

and call ``apply_os_typography(root, style)`` right after creating the
``Tk()`` root and ``ttk.Style()``. The helper sets ``TkDefaultFont``,
``TkTextFont``, ``TkHeadingFont``, ``TkCaptionFont``, ``TkSmallCaptionFont``,
``TkTooltipFont`` plus ``*Font`` option-adds and ttk ``.`` / ``TNotebook.Tab``
fonts so every current *and future* widget inherits the OS stack.

``resolve_family(preferred_list, available)`` picks the first installed
family so Linux boxes with only ``Liberation Sans`` still get a good match.
Pass a ``tk`` root (or rely on ``tkinter.font.families()``) for exact
detection; otherwise the OS primary is returned.
"""

from __future__ import annotations

import platform
import os

# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

_SYSTEM = platform.system()

if _SYSTEM == "Darwin":
    OS_TYPE = "macos"
elif _SYSTEM == "Windows":
    OS_TYPE = "windows"
elif _SYSTEM == "Linux":
    OS_TYPE = "linux"
else:  # e.g. FreeBSD, SunOS, or mocked values in tests
    OS_TYPE = "other"

# Allow explicit override for testing / screenshots:
#   WAAM_OS_OVERRIDE=windows|macos|linux
_OS_OVERRIDE = os.environ.get("WAAM_OS_OVERRIDE", "").strip().lower()
if _OS_OVERRIDE in ("macos", "darwin"):
    OS_TYPE = "macos"
elif _OS_OVERRIDE in ("windows", "win32"):
    OS_TYPE = "windows"
elif _OS_OVERRIDE in ("linux",):
    OS_TYPE = "linux"


def get_os_type() -> str:
    """Return normalised OS key: macos | windows | linux | other."""
    return OS_TYPE


# ---------------------------------------------------------------------------
# Per-OS family stacks (ordered by preference)
# ---------------------------------------------------------------------------

UI_FAMILIES: dict[str, list[str]] = {
    "macos": ["Helvetica Neue", "Helvetica", "SF Pro Text"],
    "windows": ["Segoe UI", "Tahoma", "Verdana"],
    "linux": ["DejaVu Sans", "Ubuntu", "Liberation Sans", "Noto Sans", "FreeSans"],
    "other": ["DejaVu Sans", "Liberation Sans", "Noto Sans"],
}

MONO_FAMILIES: dict[str, list[str]] = {
    "macos": ["Menlo", "SF Mono", "Monaco"],
    "windows": ["Consolas", "Cascadia Mono", "Lucida Console"],
    "linux": ["DejaVu Sans Mono", "Ubuntu Mono", "Liberation Mono", "Noto Sans Mono", "FreeMono"],
    "other": ["DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono"],
}

UI_FONT_FAMILY: str = UI_FAMILIES.get(OS_TYPE, UI_FAMILIES["other"])[0]
MONO_FONT_FAMILY: str = MONO_FAMILIES.get(OS_TYPE, MONO_FAMILIES["other"])[0]


def ui_families(os_type: str | None = None) -> list[str]:
    """Ordered UI family preferences for an OS (defaults to detected)."""
    return list(UI_FAMILIES.get(os_type or OS_TYPE, UI_FAMILIES["other"]))


def mono_families(os_type: str | None = None) -> list[str]:
    """Ordered mono family preferences for an OS (defaults to detected)."""
    return list(MONO_FAMILIES.get(os_type or OS_TYPE, MONO_FAMILIES["other"]))


def resolve_family(preferred: list[str], available: list[str] | None) -> str:
    """Pick first preferred family present in available (case-insensitive).

    Falls back to ``preferred[0]`` when no Tk context exists or nothing
    matches, so callers always get a usable Tk family name.
    """
    if not preferred:
        return "TkDefaultFont"
    if not available:
        return preferred[0]
    lowered = {a.lower(): a for a in available}
    for name in preferred:
        hit = lowered.get(name.lower())
        if hit:
            return hit  # return installed spelling
    return preferred[0]


def available_families(root=None) -> list[str]:
    """Best-effort list of installed Tk families (empty when headless)."""
    try:
        import tkinter.font as tkfont

        if root is not None:
            try:
                return list(tkfont.families(root))
            except Exception:
                pass
        try:
            return list(tkfont.families())
        except Exception:
            return []
    except Exception:
        return []


def get_ui_family(root=None, os_type: str | None = None) -> str:
    """Resolved UI family for this machine (installed match or OS primary)."""
    return resolve_family(ui_families(os_type), available_families(root))


def get_mono_family(root=None, os_type: str | None = None) -> str:
    """Resolved mono family for this machine (installed match or OS primary)."""
    return resolve_family(mono_families(os_type), available_families(root))


# ---------------------------------------------------------------------------
# Per-OS size table
# ---------------------------------------------------------------------------
# Same semantic roles on every OS; point sizes nudged so x-height and window
# fit stay comparable. DejaVu is wide -> title smaller; Segoe is tall ->
# title 1pt smaller than macOS.

_SIZE_TABLE: dict[str, dict[str, int]] = {
    "macos": {
        "title": 16, "subtitle": 10, "section": 10, "label": 10,
        "entry": 10, "console": 10, "status": 11, "btn_run": 12,
        "btn": 10, "tab": 10, "mono_sm": 9, "small": 9, "tiny": 8,
    },
    "windows": {
        "title": 15, "subtitle": 10, "section": 10, "label": 10,
        "entry": 10, "console": 10, "status": 11, "btn_run": 12,
        "btn": 10, "tab": 10, "mono_sm": 9, "small": 9, "tiny": 8,
    },
    "linux": {
        "title": 14, "subtitle": 10, "section": 10, "label": 10,
        "entry": 10, "console": 10, "status": 11, "btn_run": 12,
        "btn": 10, "tab": 10, "mono_sm": 9, "small": 9, "tiny": 8,
    },
    "other": {
        "title": 14, "subtitle": 10, "section": 10, "label": 10,
        "entry": 10, "console": 10, "status": 11, "btn_run": 12,
        "btn": 10, "tab": 10, "mono_sm": 9, "small": 9, "tiny": 8,
    },
}

_SZ = _SIZE_TABLE.get(OS_TYPE, _SIZE_TABLE["other"])

# ---------------------------------------------------------------------------
# Public Tk font tuples (backward compatible with previous gui/app.py names)
# ---------------------------------------------------------------------------

FONT_TITLE = (UI_FONT_FAMILY, _SZ["title"], "bold")
FONT_SUBTITLE = (UI_FONT_FAMILY, _SZ["subtitle"])
FONT_SECTION = (UI_FONT_FAMILY, _SZ["section"], "bold")
FONT_LABEL = (UI_FONT_FAMILY, _SZ["label"])
FONT_ENTRY = (MONO_FONT_FAMILY, _SZ["entry"])
FONT_CONSOLE = (MONO_FONT_FAMILY, _SZ["console"])
FONT_STATUS = (UI_FONT_FAMILY, _SZ["status"], "bold")
FONT_BTN_RUN = (UI_FONT_FAMILY, _SZ["btn_run"], "bold")
FONT_BTN = (UI_FONT_FAMILY, _SZ["btn"])
FONT_TAB = (UI_FONT_FAMILY, _SZ["tab"], "bold")
FONT_MONO_SM = (MONO_FONT_FAMILY, _SZ["mono_sm"])

# Helpers replacing the ~80 hardcoded ("Helvetica", 9) / ("Menlo", 9) tuples.
FONT_SMALL = (UI_FONT_FAMILY, _SZ["small"])
FONT_SMALL_BOLD = (UI_FONT_FAMILY, _SZ["small"], "bold")
FONT_UNIT = (UI_FONT_FAMILY, _SZ["small"])          # "mm", "s", "W", "[-]" suffixes
FONT_MONO = (MONO_FONT_FAMILY, _SZ["small"])        # info / status mono lines
FONT_INFO = (MONO_FONT_FAMILY, _SZ["small"])        # alias for info cards
FONT_TINY = (MONO_FONT_FAMILY, _SZ["tiny"])         # dense footnotes
FONT_COMBO = (UI_FONT_FAMILY, _SZ["small"])         # ttk.Combobox rows
# Larger variants kept OS-aware (dialog titles, code viewer).
FONT_DLG_TITLE = (UI_FONT_FAMILY, 13, "bold")
FONT_MONO_LG = (MONO_FONT_FAMILY, 11)


def fonts_for(os_type: str) -> dict[str, tuple]:
    """Font table for a given OS without importing side effects.

    Useful for previews/tests: ``fonts_for("windows")["ui"]`` etc.
    """
    ui = UI_FAMILIES.get(os_type, UI_FAMILIES["other"])[0]
    mono = MONO_FAMILIES.get(os_type, MONO_FAMILIES["other"])[0]
    sz = _SIZE_TABLE.get(os_type, _SIZE_TABLE["other"])
    return {
        "ui": ui,
        "mono": mono,
        "title": (ui, sz["title"], "bold"),
        "subtitle": (ui, sz["subtitle"]),
        "section": (ui, sz["section"], "bold"),
        "label": (ui, sz["label"]),
        "entry": (mono, sz["entry"]),
        "console": (mono, sz["console"]),
        "status": (ui, sz["status"], "bold"),
        "btn_run": (ui, sz["btn_run"], "bold"),
        "btn": (ui, sz["btn"]),
        "tab": (ui, sz["tab"], "bold"),
        "mono_sm": (mono, sz["mono_sm"]),
        "small": (ui, sz["small"]),
        "small_bold": (ui, sz["small"], "bold"),
        "tiny": (mono, sz["tiny"]),
    }


def describe() -> dict:
    """Introspection helper for logs / diagnostics."""
    return {
        "os_type": OS_TYPE,
        "platform_system": _SYSTEM,
        "ui_family": UI_FONT_FAMILY,
        "mono_family": MONO_FONT_FAMILY,
        "ui_stack": ui_families(),
        "mono_stack": mono_families(),
        "sizes": dict(_SZ),
    }


# ---------------------------------------------------------------------------
# Window application
# ---------------------------------------------------------------------------

def apply_os_typography(root, style=None, verbose: bool = False) -> dict:
    """Detect OS and apply native typography to a Tk window.

    - Resolves installed UI/mono families (falls back to OS primary).
    - Rebuilds module-level FONT_* tuples in place so already-imported
      references (``from gui.fonts import FONT_BTN``) stay valid objects?
      NOTE: tuples are immutable, so we *rebind* module globals; callers
      that did ``from gui.fonts import FONT_BTN`` before this call keep the
      old tuple. Prefer ``import gui.fonts as F`` + ``F.FONT_BTN`` or call
      this *before* building widgets (WAAMApp does exactly that).
    - Sets named Tk fonts (TkDefaultFont/TkTextFont/...) + ``*Font``
      option-adds + ttk ``.`` / ``TNotebook.Tab`` / ``TButton`` /
      ``TCombobox`` fonts so future widgets inherit automatically.
    - Applies a sane ``tk scaling`` per platform for HiDPI consistency.

    Returns the ``describe()`` dict with resolved families.
    """
    import tkinter.font as tkfont

    global UI_FONT_FAMILY, MONO_FONT_FAMILY
    global FONT_TITLE, FONT_SUBTITLE, FONT_SECTION, FONT_LABEL, FONT_ENTRY
    global FONT_CONSOLE, FONT_STATUS, FONT_BTN_RUN, FONT_BTN, FONT_TAB
    global FONT_MONO_SM, FONT_SMALL, FONT_SMALL_BOLD, FONT_UNIT
    global FONT_MONO, FONT_INFO, FONT_TINY, FONT_COMBO
    global FONT_DLG_TITLE, FONT_MONO_LG

    try:
        installed = list(tkfont.families(root))
    except Exception:
        installed = []

    ui = resolve_family(ui_families(), installed) if installed else UI_FONT_FAMILY
    mono = resolve_family(mono_families(), installed) if installed else MONO_FONT_FAMILY
    UI_FONT_FAMILY, MONO_FONT_FAMILY = ui, mono
    sz = _SIZE_TABLE.get(OS_TYPE, _SIZE_TABLE["other"])

    # Rebind public tuples (post-import `import gui.fonts` sees new values).
    FONT_TITLE = (ui, sz["title"], "bold")
    FONT_SUBTITLE = (ui, sz["subtitle"])
    FONT_SECTION = (ui, sz["section"], "bold")
    FONT_LABEL = (ui, sz["label"])
    FONT_ENTRY = (mono, sz["entry"])
    FONT_CONSOLE = (mono, sz["console"])
    FONT_STATUS = (ui, sz["status"], "bold")
    FONT_BTN_RUN = (ui, sz["btn_run"], "bold")
    FONT_BTN = (ui, sz["btn"])
    FONT_TAB = (ui, sz["tab"], "bold")
    FONT_MONO_SM = (mono, sz["mono_sm"])
    FONT_SMALL = (ui, sz["small"])
    FONT_SMALL_BOLD = (ui, sz["small"], "bold")
    FONT_UNIT = (ui, sz["small"])
    FONT_MONO = (mono, sz["small"])
    FONT_INFO = (mono, sz["small"])
    FONT_TINY = (mono, sz["tiny"])
    FONT_COMBO = (ui, sz["small"])
    FONT_DLG_TITLE = (ui, 13, "bold")
    FONT_MONO_LG = (mono, 11)

    # Named Tk fonts so option-add inheritance works even for widgets that
    # were created with string font names.
    try:
        for name, spec in (
            ("TkDefaultFont", (ui, sz["label"])),
            ("TkTextFont", (ui, sz["label"])),
            ("TkHeadingFont", (ui, sz["section"], "bold")),
            ("TkCaptionFont", (ui, sz["btn"], "bold")),
            ("TkSmallCaptionFont", (ui, sz["small"])),
            ("TkTooltipFont", (ui, sz["small"])),
            ("TkIconFont", (ui, sz["small"])),
            ("TkMenuFont", (ui, sz["small"])),
        ):
            try:
                f = tkfont.nametofont(name, root)
                f.configure(family=spec[0], size=spec[1])
                if len(spec) > 2:
                    f.configure(weight=spec[2])
            except Exception:
                continue
    except Exception:
        pass

    # Catch-all defaults for widgets created without explicit font=.
    try:
        root.option_add("*Font", (ui, sz["label"]))
        root.option_add("*Text.Font", (mono, sz["entry"]))
        root.option_add("*Entry.Font", (mono, sz["entry"]))
    except Exception:
        pass

    # ttk inherits from "."; keep tab font OS-aware too.
    try:
        if style is not None:
            style.configure(".", font=(ui, sz["label"]))
            style.configure("TButton", font=(ui, sz["btn"]))
            style.configure("TCombobox", font=(ui, sz["small"]))
            style.configure("TNotebook.Tab", font=(ui, sz["tab"], "bold"))
            style.configure("Treeview", font=(ui, sz["small"]))
    except Exception:
        pass

    # Platform Tk scaling: macOS Retina is already scaled; Windows needs
    # ~1.0-1.25 depending on DPI; Linux Xft varies. Keep conservative so the
    # fixed 1300x760 geometry does not blow up.
    try:
        if OS_TYPE == "windows":
            root.tk.call("tk", "scaling", 1.1)
        elif OS_TYPE == "linux":
            root.tk.call("tk", "scaling", 1.0)
        else:
            root.tk.call("tk", "scaling", 1.0)
    except Exception:
        pass

    info = describe()
    info["resolved_ui"] = ui
    info["resolved_mono"] = mono
    if verbose:
        print(f"[FONTS] OS={OS_TYPE} UI={ui} Mono={mono} sizes={sz}")
    return info

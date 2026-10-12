#!/usr/bin/env python3
"""
WAAM Multi-Physics Simulation Framework — GUI Application

A Tkinter-based desktop interface for configuring and running
FreeFEM++ WAAM simulations with multi-physics coupling.
Features:
  - Default Light Theme with instantaneous Light/Dark Theme toggle
  - Vertical layout: Configuration Notebook on Left, Tall Live Solver Stream on Right
  - Direct structured mesh sizing inputs (dx, dy, dz in mm) + Live Element Size Readout
  - 6 selectable heat source models with dynamic parameter panels
  - Material database with T-dependent property auto-fill
  - Slicing & CAD toolpath integration
  - Real-time FreeFEM++ console output stream
  - OpenFOAM 2412 and ParaView one-click integration
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import json
import os
import sys
import re
import datetime
import shutil
import glob
import math
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from codegen.edp_builder import EDPBuilder


class Tooltip:
    """Minimal hover tooltip (no dependency). Usage: Tooltip(widget, text)."""

    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tip = None
        self._job = None
        try:
            widget.bind("<Enter>", self._schedule, add="+")
            widget.bind("<Leave>", self._hide, add="+")
            widget.bind("<ButtonPress>", self._hide, add="+")
        except Exception:
            pass

    def _schedule(self, _event=None):
        try:
            self._hide()
            self._job = self.widget.after(self.delay, self._show)
        except Exception:
            pass

    def _show(self):
        if self._tip is not None:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self._tip = tk.Toplevel(self.widget)
            self._tip.wm_overrideredirect(True)
            self._tip.wm_geometry(f"+{x}+{y}")
            lbl = tk.Label(
                self._tip, text=self.text, justify=tk.LEFT,
                bg="#111827", fg="#f9fafc", font=("TkDefaultFont", 9),
                padx=8, pady=5, wraplength=320,
            )
            lbl.pack()
        except Exception:
            pass

    def _hide(self, _event=None):
        try:
            if self._job is not None:
                self.widget.after_cancel(self._job)
        except Exception:
            pass
        self._job = None
        try:
            if self._tip is not None:
                self._tip.destroy()
        except Exception:
            pass
        self._tip = None


# ============================================================
# THEME PALETTES (Light & Dark)
# ============================================================
THEMES = {
    "light": {
        "bg": "#f8f9fa",
        "bg_card": "#ffffff",
        "bg_input": "#ffffff",
        "fg": "#000000",
        "fg_dim": "#475569",
        "accent": "#0d6efd",
        "accent2": "#198754",
        "warning": "#d97706",
        "error": "#dc3545",
        "border": "#cbd5e1",
        "button": "#e2e8f0",
        "button_fg": "#000000",
        "button_run": "#10b981",
        "button_run_fg": "#000000",
        "button_primary": "#2563eb",
        "button_primary_fg": "#000000",
        "button_primary_active": "#1d4ed8",
        "button_violet": "#7c3aed",
        "button_violet_fg": "#000000",
        "button_violet_active": "#6d28d9",
        "button_info": "#0284c7",
        "button_info_fg": "#000000",
        "tab_selected_bg": "#0d6efd",
        "tab_selected_fg": "#ffffff",
        "tab_unselected_bg": "#e2e8f0",
        "tab_unselected_fg": "#000000",
        "console_bg": "#1e293b",
        "console_fg": "#f1f5f9"
    },
    "dark": {
        "bg": "#1e1e2e",
        "bg_card": "#2a2a3d",
        "bg_input": "#363650",
        "fg": "#cdd6f4",
        "fg_dim": "#7f849c",
        "accent": "#89b4fa",
        "accent2": "#a6e3a1",
        "warning": "#f9e2af",
        "error": "#f38ba8",
        "border": "#45475a",
        "button": "#45475a",
        "button_fg": "#cdd6f4",
        "button_run": "#a6e3a1",
        "button_run_fg": "#1e1e2e",
        "button_primary": "#89b4fa",
        "button_primary_fg": "#1e1e2e",
        "button_primary_active": "#a5c4fc",
        "button_violet": "#c4b5fd",
        "button_violet_fg": "#1e1e2e",
        "button_violet_active": "#a78bfa",
        "button_info": "#38bdf8",
        "button_info_fg": "#0f172a",
        "tab_selected_bg": "#89b4fa",
        "tab_selected_fg": "#1e1e2e",
        "tab_unselected_bg": "#2a2a3d",
        "tab_unselected_fg": "#cdd6f4",
        "console_bg": "#11111b",
        "console_fg": "#cdd6f4"
    }
}


MM_DIMENSION_KEYS = {
    "lx", "ly_base", "hz_base", "ty_dep", "h_bead", "h_bead_first", "h_bead_subseq",
    "d_wire", "af", "ar", "bG", "cG",
    "R_beam", "R_spot", "rt", "rb", "d_pen",
    "a_band", "b_band", "c_band", "R_activate"
}
MM_SPEED_KEYS = {"v_scan"}


# Font definitions — OS-aware native stacks (see gui/fonts.py).
# Detects macOS / Windows / Linux and picks Helvetica Neue-Menlo,
# Segoe UI-Consolas, or DejaVu Sans(-Mono) with per-OS size nudges.
# Imported names stay backward compatible; apply_os_typography() rebinds
# them to installed families at window creation.
try:
    from gui.fonts import (
        FONT_TITLE, FONT_SUBTITLE, FONT_SECTION, FONT_LABEL, FONT_ENTRY,
        FONT_CONSOLE, FONT_STATUS, FONT_BTN_RUN, FONT_BTN, FONT_TAB,
        FONT_MONO_SM, FONT_SMALL, FONT_SMALL_BOLD, FONT_UNIT,
        FONT_MONO, FONT_INFO, FONT_TINY, FONT_COMBO,
        FONT_DLG_TITLE, FONT_MONO_LG,
        OS_TYPE, UI_FONT_FAMILY, MONO_FONT_FAMILY,
        apply_os_typography,
    )
except ImportError:  # fallback when run as script without package context
    try:
        from fonts import (
            FONT_TITLE, FONT_SUBTITLE, FONT_SECTION, FONT_LABEL, FONT_ENTRY,
            FONT_CONSOLE, FONT_STATUS, FONT_BTN_RUN, FONT_BTN, FONT_TAB,
            FONT_MONO_SM, FONT_SMALL, FONT_SMALL_BOLD, FONT_UNIT,
            FONT_MONO, FONT_INFO, FONT_TINY, FONT_COMBO,
            FONT_DLG_TITLE, FONT_MONO_LG,
            OS_TYPE, UI_FONT_FAMILY, MONO_FONT_FAMILY,
            apply_os_typography,
        )
    except ImportError:
        OS_TYPE = "other"
        UI_FONT_FAMILY = "Helvetica Neue"
        MONO_FONT_FAMILY = "Menlo"
        FONT_TITLE   = ("Helvetica Neue", 16, "bold")
        FONT_SUBTITLE= ("Helvetica Neue", 10)
        FONT_SECTION = ("Helvetica Neue", 10, "bold")
        FONT_LABEL   = ("Helvetica Neue", 10)
        FONT_ENTRY   = ("Menlo", 10)
        FONT_CONSOLE = ("Menlo", 10)
        FONT_STATUS  = ("Helvetica Neue", 11, "bold")
        FONT_BTN_RUN = ("Helvetica Neue", 12, "bold")
        FONT_BTN     = ("Helvetica Neue", 10)
        FONT_TAB     = ("Helvetica Neue", 10, "bold")
        FONT_MONO_SM = ("Menlo", 9)
        FONT_SMALL = ("Helvetica Neue", 9)
        FONT_SMALL_BOLD = ("Helvetica Neue", 9, "bold")
        FONT_UNIT = ("Helvetica Neue", 9)
        FONT_MONO = ("Menlo", 9)
        FONT_INFO = ("Menlo", 9)
        FONT_TINY = ("Menlo", 8)
        FONT_COMBO = ("Helvetica Neue", 9)
        FONT_DLG_TITLE = ("Helvetica Neue", 13, "bold")
        FONT_MONO_LG = ("Menlo", 11)
        def apply_os_typography(root, style=None, verbose=False):
            return {}


# Process parameter presets
PARAMETER_PRESETS = {
    "DEFAULT": {
        "material_base": "SS316L", "material_dep": "SS308L",
        "I_weld": 200.0, "U_weld": 24.0, "eta_arc": 0.80, "v_scan": 10.0,
        "t_wait": 60.0,
        "lx": 150.0, "ly_base": 150.0, "hz_base": 5.0,
        "ty_dep": 8.0, "h_bead": 4.0, "remelt_ratio": 0.80, "n_layers": 26,
        "heat_source": "goldak",
        "af": 8.0, "ar": 20.0, "bG": 7.0, "cG": 4.0, "ff": 0.60, "fr": 1.40,
        "nx_base": 30, "ny_base": 20, "nz_base": 8,
        "nx_dep": 30, "ny_dep": 6, "nz_dep": 26,
        "mesh_source": "parametric",
        "theta": 1.0,
        "dt_weld": 0.333, "dt_wait": 10.0, "dt_cool": 120.0,
        "n_cool_steps": 20,
        "rho": 7850.0, "T_sol": 1402.0, "T_liq": 1457.0, "latent_heat": 270000.0,
        "emissivity": 0.65, "h_conv": 15.0, "h_bot": 50.0, "nu": 0.30, "T_mech": 800.0,
        "chi_x": 1.20, "chi_y": 0.40, "chi_z": 0.00,
        "mechanical_bc": "isostatic",
        # Fixture strength is expressed as a multiple of the plate bending
        # stiffness; the builder derives k_spring_z/xy from it. See
        # EDPBuilder._derive_fixture_stiffness().
        "fixture_stiffness_ratio_z": 1.0e5, "fixture_stiffness_ratio_xy": 1.0e5,
        "enable_radiation": True, "enable_mechanical": True,
        "enable_microstructure": True, "enable_boiling_cap": False,
        "enable_yield_truncation": False,
    },
    "SS308L on SS316L (UVA-WAAM Multi-Material)": {
        "material_base": "SS316L", "material_dep": "SS308L",
        "I_weld": 180.0, "U_weld": 22.0, "eta_arc": 0.82, "v_scan": 8.0,
        "v_wire": 5.0, "d_wire": 1.2,
        "t_wait": 30.0,
        "lx": 150.0, "ly_base": 150.0, "hz_base": 5.0,
        "ty_dep": 5.0, "h_bead": 4.0, "remelt_ratio": 0.80, "n_layers": 33,
        "heat_source": "goldak",
        "af": 5.0, "ar": 12.0, "bG": 3.5, "cG": 3.5, "ff": 0.60, "fr": 1.40,
        "enable_ultrasonic": True,
        "f_US": 20000.0, "A_US": 15.0, "beta_US": 0.20,
        "k_stream": 45.0,
    },
    "SS316L Stainless Steel (GMAW-WAAM)": {
        "material_base": "SS316L", "material_dep": "SS316L",
        "I_weld": 220.0, "U_weld": 25.0, "eta_arc": 0.80, "v_scan": 8.0,
        "v_wire": 5.5, "d_wire": 1.2,
        "lx": 150.0, "ly_base": 80.0, "hz_base": 12.0,
        "ty_dep": 7.0, "h_bead": 4.0, "remelt_ratio": 0.80, "n_layers": 20,
        "heat_source": "goldak",
        "af": 7.0, "ar": 18.0, "bG": 6.0, "cG": 4.0, "ff": 0.60, "fr": 1.40,
    },
    "SS308L Stainless Steel (GMAW-WAAM)": {
        "material_base": "SS308L", "material_dep": "SS308L",
        "I_weld": 200.0, "U_weld": 24.0, "eta_arc": 0.80, "v_scan": 10.0,
        "v_wire": 5.0, "d_wire": 1.2,
        "lx": 200.0, "ly_base": 100.0, "hz_base": 10.0,
        "ty_dep": 8.0, "h_bead": 4.0, "remelt_ratio": 0.80, "n_layers": 26,
        "heat_source": "goldak",
        "af": 8.0, "ar": 20.0, "bG": 7.0, "cG": 4.0, "ff": 0.60, "fr": 1.40,
    },
    "ER70S6 Mild Steel (P48 Pragmatic GMAW-WAAM)": {
        "material_base": "MildSteel", "material_dep": "ER70S-6",
        "I_weld": 150.0, "U_weld": 20.0, "eta_arc": 0.80, "v_scan": 7.0,
        "v_wire": 4.5, "d_wire": 1.2,
        "lx": 150.0, "ly_base": 100.0, "hz_base": 6.0,
        "ty_dep": 6.0, "h_bead": 3.0, "remelt_ratio": 0.80, "n_layers": 20,
        "heat_source": "p48_surface",
        "eta_drop": 0.25, "T_drop": 1627.0,
        "enable_radiation": True, "enable_latent": True,
        "enable_mechanical": True,
    },
    "Ti-6Al-4V Titanium (Laser-DED / WAAM)": {
        "material_base": "Ti-6Al-4V", "material_dep": "Ti-6Al-4V",
        "I_weld": 1500.0, "U_weld": 1.0, "eta_arc": 0.40, "v_scan": 15.0,
        "v_wire": 3.0, "d_wire": 1.2,
        "lx": 100.0, "ly_base": 60.0, "hz_base": 8.0,
        "ty_dep": 5.0, "h_bead": 2.5, "remelt_ratio": 0.80, "n_layers": 30,
        "heat_source": "gaussian_vol",
    },
    "Pure Al (Al1xxx Wire-DED)": {
        "material_base": "Pure Al (Al1xxx)", "material_dep": "Pure Al (Al1xxx)",
        "I_weld": 140.0, "U_weld": 18.0, "eta_arc": 0.85, "v_scan": 12.0,
        "v_wire": 6.0, "d_wire": 1.2,
        "lx": 160.0, "ly_base": 90.0, "hz_base": 8.0,
        "ty_dep": 6.0, "h_bead": 3.0, "remelt_ratio": 0.80, "n_layers": 20,
        "heat_source": "goldak",
        "af": 6.0, "ar": 15.0, "bG": 5.0, "cG": 3.5, "ff": 0.60, "fr": 1.40,
    },
    # "Custom" is not a hardcoded parameter set: selecting it keeps the
    # current UI values verbatim and bypasses every preset override.
    # Any manual parameter edit auto-switches the preset selector to
    # "Custom" so the GUI never claims DEFAULT (or any other preset)
    # while the solver runs diverged values.
    "Custom": {},
}


class WAAMApp:
    """Main application window with dynamic theme switching and vertical solver layout."""

    def __init__(self, initial_preset=None):
        self.root = tk.Tk()
        self.root.title("WAAM Multi-Physics Simulator — FreeFEM++ and OpenFOAM 2412")
        # OS-aware typography: resolve native UI/mono families before any
        # widget is built so all font= references use the right stack.
        try:
            self.font_info = apply_os_typography(self.root)
        except Exception:
            self.font_info = {}
        # 1520x880 needs a 1600x900+ screen. The old minimum (1400x780) was
        # larger than a 1366x768 panel, where the window then opened
        # off-screen and the app was unusable.
        self.root.geometry("1360x800")
        self.root.minsize(1180, 700)
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            if sw < 1450 or sh < 850:
                self.root.geometry("%dx%d" % (max(sw - 40, 1000), max(sh - 90, 620)))
        except Exception:
            pass

        # Theme state: defaults to 'light' per user preference
        self.current_theme = "light"
        self.colors = THEMES[self.current_theme]

        # Simulation state
        self.params = EDPBuilder.default_params()
        self.process = None
        self.running = False
        self.active_run_dir = None
        self.active_run_executed = False
        self._param_vars = {}
        self.param_entries = {}
        self.registered_widgets = []
        self._applying_preset = False

        # Load material database
        self.materials = {}
        db_path = Path(__file__).parent.parent / "core" / "multimaterial" / "material_database.json"
        if db_path.exists():
            try:
                with open(db_path) as f:
                    self.materials = json.load(f).get("alloys", {})
            except Exception:
                pass

        self._build_ui()
        preset_to_load = initial_preset if (initial_preset and initial_preset in PARAMETER_PRESETS) else "DEFAULT"
        if preset_to_load in PARAMETER_PRESETS:
            self.preset_var.set(preset_to_load)
            self._apply_preset()
        self._apply_theme()
        self._update_mesh_calc()
        self._update_heat_calc()
        def _on_tab_changed(event=None):
            self._refresh_embedded_viewport()
            try:
                selected_tab = self.notebook.select()
                if hasattr(self, "default_case_tab") and selected_tab == str(self.default_case_tab):
                    self._refresh_generated_edp_preview()
            except Exception:
                pass
        self.notebook.bind("<<NotebookTabChanged>>", _on_tab_changed)
        # Keyboard shortcuts: Cmd/Ctrl+Enter runs, Esc stops, Ctrl/Cmd+F finds in log.
        try:
            self.root.bind_all("<Control-Return>", lambda e: self._run_simulation())
            self.root.bind_all("<Command-Return>", lambda e: self._run_simulation())
            self.root.bind_all("<Escape>", lambda e: self._stop_simulation())
        except Exception:
            pass

    def _build_ui(self):
        """Build UI structure with horizontal split (Tabs on Left, Live Solver on Right)."""
        # Fix macOS Copy/Paste bindings
        if self.root.tk.call('tk', 'windowingsystem') == 'aqua':
            self.root.bind_all('<Command-c>', lambda e: e.widget.event_generate('<<Copy>>'))
            self.root.bind_all('<Command-v>', lambda e: e.widget.event_generate('<<Paste>>'))
            self.root.bind_all('<Command-x>', lambda e: e.widget.event_generate('<<Cut>>'))
            self.root.bind_all('<Command-a>', lambda e: e.widget.event_generate('<<SelectAll>>'))

        self.style = ttk.Style()
        self.style.theme_use("clam")
        # Re-apply with style so ttk "." / TNotebook.Tab / TButton inherit OS fonts.
        try:
            self.font_info = apply_os_typography(self.root, self.style)
        except Exception:
            pass
        self.style.configure("TNotebook.Tab", padding=[8, 3], font=FONT_TAB)

        # Top Header Bar
        self.header = tk.Frame(self.root)
        self.header.pack(fill=tk.X, padx=14, pady=(8, 4))

        self.title_frame = tk.Frame(self.header)
        self.title_frame.pack(side=tk.LEFT)
        self.title_label = tk.Label(
            self.title_frame, text="WAAM Multi-Physics Simulator",
            font=FONT_TITLE
        )
        self.title_label.pack(side=tk.TOP, anchor=tk.W)
        # Subtitle removed per user request

        # Header Right: Preset selector + Status + Theme Toggle
        self.header_right = tk.Frame(self.header)
        self.header_right.pack(side=tk.RIGHT)

        self.theme_btn = tk.Button(
            self.header_right, text="[ Dark Theme ]",
            font=FONT_BTN, relief=tk.GROOVE,
            command=self._toggle_theme, padx=10, pady=3
        )
        self.theme_btn.pack(side=tk.RIGHT, padx=6)

        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = tk.Label(
            self.header_right, textvariable=self.status_var,
            font=FONT_STATUS
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Preset Selector
        preset_frame = tk.Frame(self.header_right)
        preset_frame.pack(side=tk.RIGHT, padx=12)
        self._reg(preset_frame, "card_frame")
        tk.Label(preset_frame, text="Preset:", font=FONT_BTN).pack(side=tk.LEFT, padx=(0, 4))
        self._reg(preset_frame.winfo_children()[0], "label")
        self.preset_var = tk.StringVar(value="DEFAULT")
        preset_names = list(PARAMETER_PRESETS.keys())
        self.preset_combo = ttk.Combobox(
            preset_frame, textvariable=self.preset_var,
            values=preset_names, width=30, state="readonly", font=FONT_BTN
        )
        self.preset_combo.pack(side=tk.LEFT)
        self.preset_combo.bind("<<ComboboxSelected>>", self._apply_preset)

        # Main Paned Area: LEFT (Tabs) + RIGHT (Vertical Live Solver & Actions)
        self.main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Left Notebook (Parameters & 3D Viewport)
        self.notebook = ttk.Notebook(self.main_pane)
        self.main_pane.add(self.notebook, width=980, minsize=900)

        self._build_geometry_tab()
        self._build_process_tab()
        self._build_material_tab()
        self._build_mesh_cad_tab()
        self._build_physics_tab()
        self._build_solver_tab()
        self._build_openfoam_tab()
        self._build_parametric_tab()
        self._build_viewport_tab()
        self._build_default_case_tab()
        self._build_claude_tab()
        self._build_sim_log_tab()

        # File menu bar
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Project (.waam)...", command=self._save_project)
        file_menu.add_command(label="Load Project (.waam)...", command=self._load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Open Output Folder", command=self._open_output)
        file_menu.add_command(label="Open Report", command=self._open_report)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._confirm_quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Multi-Material & Ultrasonic menu
        claude_menu = tk.Menu(menubar, tearoff=0)
        claude_menu.add_command(label="Open Ultrasonic & Multi-Material Tab", command=lambda: self.notebook.select(self.claude_tab))
        claude_menu.add_separator()
        claude_menu.add_command(label="Open Results in ParaView", command=self._open_paraview)
        claude_menu.add_command(label="Open Output Directory", command=self._open_output)
        menubar.add_cascade(label="Ultrasonic", menu=claude_menu)

        self.root.config(menu=menubar)

        # Right Panel: Action Toolbar + Tall Vertical Live Solver Console
        self.right_panel = tk.Frame(self.main_pane)
        self.main_pane.add(self.right_panel, width=540, minsize=420)
        self._reg(self.right_panel, "card_frame")

        # Top Action Button Bar inside Right Panel
        self.btn_bar = tk.Frame(self.right_panel, pady=4)
        self.btn_bar.pack(fill=tk.X, padx=6, pady=(2, 4))
        self._reg(self.btn_bar, "card_frame")

        # Main Run & Stop Buttons
        run_stop_frame = tk.Frame(self.btn_bar)
        run_stop_frame.pack(fill=tk.X, pady=(0, 4))
        self._reg(run_stop_frame, "card_frame")

        self.run_btn = tk.Button(
            run_stop_frame, text="▶ Run Simulation",
            font=FONT_BTN_RUN, relief=tk.FLAT, bd=0, highlightthickness=0,
            command=self._run_simulation, padx=16, pady=6
        )
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.stop_btn = tk.Button(
            run_stop_frame, text="⏹ Stop",
            font=FONT_BTN, relief=tk.FLAT, bd=0, highlightthickness=0,
            command=self._stop_simulation, padx=12, pady=6,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.RIGHT, padx=(4, 0))

        # Primary Results & Post-Processing Bar
        res_bar = tk.Frame(self.btn_bar)
        res_bar.pack(fill=tk.X, pady=(2, 2))
        self._reg(res_bar, "card_frame")

        self.view_results_btn = tk.Button(
            res_bar, text="📄 View Report",
            font=FONT_BTN, relief=tk.FLAT, bd=0, highlightthickness=0,
            bg=self.colors["button_primary"], fg=self.colors["button_primary_fg"],
            activebackground=self.colors["button_primary_active"],
            activeforeground=self.colors["button_primary_fg"],
            command=self._open_report, padx=10, pady=4
        )
        self.view_results_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.report_btn = self.view_results_btn

        self.pv_btn = tk.Button(
            res_bar, text="📊 ParaView", font=FONT_BTN, relief=tk.FLAT, bd=0, highlightthickness=0,
            command=self._open_paraview, padx=8, pady=4
        )
        self.pv_btn.pack(side=tk.LEFT, padx=2)

        self.out_btn = tk.Button(
            res_bar, text="📁 Run Folder", font=FONT_BTN, relief=tk.FLAT, bd=0, highlightthickness=0,
            command=self._open_output, padx=8, pady=4
        )
        self.out_btn.pack(side=tk.LEFT, padx=(2, 0))

        # Secondary Modeling & Case Generation Bar (2 balanced buttons, no truncation!)
        sec_bar = tk.Frame(self.btn_bar)
        sec_bar.pack(fill=tk.X, pady=(2, 2))
        self._reg(sec_bar, "card_frame")

        self.gen_btn = tk.Button(
            sec_bar, text="⚡ Generate Simulation Script", font=FONT_BTN, relief=tk.FLAT, bd=0, highlightthickness=0,
            command=self._generate_script, padx=6, pady=4
        )
        self.gen_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.foam_btn = tk.Button(
            sec_bar, text="🌊 Generate CFD Case", font=FONT_BTN, relief=tk.FLAT, bd=0, highlightthickness=0,
            command=self._openfoam_dialog, padx=6, pady=4
        )
        self.foam_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # Viewport & Multi-Material Toolbar Row (3 buttons, cleanly distributed!)
        vis_bar = tk.Frame(self.btn_bar)
        vis_bar.pack(fill=tk.X, pady=(2, 2))
        self._reg(vis_bar, "card_frame")

        self.mesh3d_btn = tk.Button(
            vis_bar, text="🌐 Show 3D Tab", font=FONT_BTN, relief=tk.FLAT, bd=0, highlightthickness=0,
            bg=self.colors["button_info"], fg=self.colors["button_info_fg"],
            command=self._open_3d_mesh_viewer, padx=6, pady=4
        )
        self.mesh3d_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.pv_gpu_btn = tk.Button(
            vis_bar, text="🧊 Open 3D Window", font=FONT_BTN, relief=tk.FLAT, bd=0, highlightthickness=0,
            command=self._open_pyvista_window, padx=6, pady=4
        )
        self.pv_gpu_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.claude_btn = tk.Button(
            vis_bar, text="🔬 Ultrasonic Tab", font=FONT_BTN, relief=tk.FLAT, bd=0, highlightthickness=0,
            bg=self.colors["button_violet"], fg=self.colors["button_violet_fg"],
            activebackground=self.colors["button_violet_active"],
            activeforeground=self.colors["button_violet_fg"],
            command=lambda: self.notebook.select(self.claude_tab), padx=6, pady=4
        )
        self.claude_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        self.claude_pv_btn = self.pv_btn

        # Live Simulation Stage & Progress Tracker
        self.tracker_frame = tk.LabelFrame(
            self.right_panel, text="Simulation Progress",
            font=FONT_SECTION, padx=8, pady=4
        )
        self.tracker_frame.pack(fill=tk.X, padx=6, pady=(2, 4))
        self._reg(self.tracker_frame, "card_labelframe")
        self._add_info_btn_to_labelframe(self.tracker_frame, self._show_physics_info)

        self.progress_bar = ttk.Progressbar(self.tracker_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=4, pady=(2, 3))

        self.stage_label = tk.Label(
            self.tracker_frame, text="Ready to Run",
            font=FONT_SECTION, fg="#0d6efd"
        )
        self.stage_label.pack(anchor=tk.W, padx=4)
        self._reg(self.stage_label, "label")

        self.step_detail_label = tk.Label(
            self.tracker_frame, text="Progress: 0%  |  Peak Temp: 25.0 C",
            font=FONT_MONO_SM, fg="#64748b"
        )
        self.step_detail_label.pack(anchor=tk.W, padx=4)
        self._reg(self.step_detail_label, "label_dim")

        # Full-height Vertical Console Frame
        self.console_frame = tk.LabelFrame(
            self.right_panel, text="Solver Output Stream",
            font=FONT_SECTION
        )
        self.console_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))
        self._reg(self.console_frame, "card_labelframe")
        self._add_info_btn_to_labelframe(self.console_frame, self._show_physics_info)

        # Sleek mini-toolbar with live status, auto-scroll toggle, copy and clear
        self.console_toolbar = tk.Frame(self.console_frame, bg=self.colors["bg_card"])
        self.console_toolbar.pack(fill=tk.X, padx=6, pady=(2, 4))
        self._reg(self.console_toolbar, "card_frame")

        self.console_status_dot = tk.Label(
            self.console_toolbar, text="● IDLE",
            font=FONT_TINY, fg="#64748b", bg=self.colors["bg_card"]
        )
        self.console_status_dot.pack(side=tk.LEFT, padx=(2, 8))
        self._reg(self.console_status_dot, "label_dim")

        self.console_autoscroll_var = tk.BooleanVar(value=True)
        self.console_autoscroll_chk = ttk.Checkbutton(
            self.console_toolbar, text="Auto-Scroll",
            variable=self.console_autoscroll_var
        )
        self.console_autoscroll_chk.pack(side=tk.LEFT, padx=4)

        self.console_copy_btn = tk.Button(
            self.console_toolbar, text="Copy Output",
            font=FONT_SMALL, relief=tk.FLAT, bd=0, padx=6, pady=2,
            bg=self.colors["button"], fg=self.colors["button_fg"],
            command=self._console_copy
        )
        self.console_copy_btn.pack(side=tk.RIGHT, padx=2)
        self._reg(self.console_copy_btn, "button")

        self.console_clear_btn = tk.Button(
            self.console_toolbar, text="Clear Stream",
            font=FONT_SMALL, relief=tk.FLAT, bd=0, padx=6, pady=2,
            bg=self.colors["button"], fg=self.colors["button_fg"],
            command=self._console_clear
        )
        self.console_clear_btn.pack(side=tk.RIGHT, padx=2)
        self._reg(self.console_clear_btn, "button")

        self.console = scrolledtext.ScrolledText(
            self.console_frame,
            font=FONT_CONSOLE, wrap=tk.WORD, state=tk.DISABLED,
            spacing1=3, spacing2=1, spacing3=3,
            padx=12, pady=8
        )
        self.console.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        self._init_console_tags()
        self._add_tooltips()

    def _add_tooltips(self):
        """Hover help for jargon fields and ambiguous actions."""
        try:
            Tooltip(self.run_btn, "Run FreeFEM simulation for the active case.\nShortcut: Ctrl/Cmd+Enter")
            Tooltip(self.stop_btn, "Stop the running simulation.\nShortcut: Esc. Keeps partial results.")
            Tooltip(self.view_results_btn, "Open the HTML report for the active run folder.")
            Tooltip(self.pv_btn, "Open the active run (FreeFEM visual outputs) in ParaView.")
            Tooltip(self.out_btn, "Reveal the active timestamped run folder in Finder/Explorer.")
            Tooltip(self.gen_btn, "Generate waam_simulation.edp + parameters.json without solving.")
            Tooltip(self.mesh3d_btn, "Switch to the embedded 3D Viewport tab.")
            Tooltip(self.pv_gpu_btn, "Open an isolated PyVista 3D window (avoids Tk/VTK event clash).")
            Tooltip(self.foam_btn, "Generate a BeamWeldFoam CFD case from Process + Material tabs.")
            Tooltip(self.claude_btn, "Open custom multi-material stack configuration.")
            Tooltip(self.claude_100mm_btn, "Run the 50-layer 100 mm multi-material wall.")
            Tooltip(self.claude_pv_btn, "Open multi-material results (waam.pvd) in ParaView.")
            Tooltip(self.theme_btn, "Toggle light/dark theme. All panels follow.")
            Tooltip(self.console_clear_btn, "Clear the active solver output stream.")
            Tooltip(self.console_copy_btn, "Copy all solver output text to clipboard.")
            Tooltip(self.console_autoscroll_chk, "Toggle auto-scrolling to the bottom as new lines stream in.")
        except Exception:
            pass

    def _build_openfoam_tab(self):
        """Dedicated OpenFOAM CFD (beamWeldFoam) tab with melt pool simulation controls."""
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="OpenFOAM CFD")
        self._reg(tab, "card_frame")

        # Scrollable container
        canvas = tk.Canvas(tab, borderwidth=0, highlightthickness=0,
                           bg=self.colors["bg_card"], highlightbackground=self.colors["border"])
        self.openfoam_canvas = canvas
        v_scroll = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_content = tk.Frame(canvas)
        self._reg(scroll_content, "card_frame")
        scroll_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.configure(yscrollcommand=v_scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # --- CFD Solver Info ---
        info_lf = tk.LabelFrame(scroll_content, text="beamWeldFoam -- VOF Melt Pool CFD (OpenFOAM 2412)",
                                font=FONT_SECTION, padx=8, pady=4)
        info_lf.pack(fill=tk.X, padx=8, pady=3)
        self._reg(info_lf, "card_labelframe")
        self._add_info_btn_to_labelframe(info_lf, self._show_openfoam_info)
        info_text = (
            "Two-phase VOF solver with diverging-Gaussian heat source, Marangoni\n"
            "convection, vapor recoil, Darcy mushy damping, and latent heat.\n"
            "Local melt-pool window (X transverse / Y vertical / Z travel, ms\n"
            "timescale). Arc power, speed, and material auto-populate from the\n"
            "Process and Material tabs; window auto-sizes from CAD/G-code bbox."
        )
        info_lbl = tk.Label(info_lf, text=info_text, font=FONT_MONO, justify=tk.LEFT, anchor=tk.W)
        info_lbl.pack(fill=tk.X, padx=4, pady=2)
        self._reg(info_lbl, "label_dim")

        # --- Time Controls ---
        lf_time = tk.LabelFrame(scroll_content, text="CFD Time Controls",
                                font=FONT_SECTION, padx=8, pady=4)
        lf_time.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_time, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_time, self._show_openfoam_info)

        # Millisecond VOF regime defaults (port-validated). A seconds-scale
        # endTime at deltaT=1e-7 would need tens of millions of steps.
        self.cfd_endtime_var = tk.StringVar(value="0.0005")
        self.cfd_deltat_var = tk.StringVar(value="1e-7")
        self.cfd_writeint_var = tk.StringVar(value="5e-5")
        self.cfd_maxco_var = tk.StringVar(value="0.2")
        self.cfd_maxalphaco_var = tk.StringVar(value="0.2")
        self.cfd_maxdeltat_var = tk.StringVar(value="1e-5")

        cfd_time_fields = [
            ("End Time:", self.cfd_endtime_var, "s"),
            ("Initial deltaT:", self.cfd_deltat_var, "s"),
            ("Write Interval:", self.cfd_writeint_var, "s"),
            ("Max Courant No.:", self.cfd_maxco_var, "[-]"),
            ("Max Alpha Courant:", self.cfd_maxalphaco_var, "[-]"),
            ("Max deltaT:", self.cfd_maxdeltat_var, "s"),
        ]
        for i, (label, var, unit) in enumerate(cfd_time_fields):
            lbl = tk.Label(lf_time, text=label, font=FONT_LABEL)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=(8, 4), pady=2)
            self._reg(lbl, "label")
            entry = tk.Entry(lf_time, textvariable=var, width=12, font=FONT_ENTRY, relief=tk.SOLID, bd=1)
            entry.grid(row=i, column=1, padx=4, pady=2, sticky=tk.W)
            self._reg(entry, "entry")
            u_lbl = tk.Label(lf_time, text=unit, font=FONT_SMALL)
            u_lbl.grid(row=i, column=2, sticky=tk.W, padx=4)
            self._reg(u_lbl, "label_dim")

        # --- Mesh Resolution ---
        lf_mesh = tk.LabelFrame(scroll_content, text="CFD Mesh Resolution",
                                font=FONT_SECTION, padx=8, pady=4)
        lf_mesh.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_mesh, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_mesh, self._show_openfoam_info)

        self.cfd_cellsize_var = tk.StringVar(value="1.0")
        self.cfd_atmoheight_var = tk.StringVar(value="3.0")
        self.cfd_travel_var = tk.StringVar(value="40.0")
        self.cfd_width_var = tk.StringVar(value="auto")

        mesh_fields = [
            ("Target Cell Size:", self.cfd_cellsize_var, "mm"),
            ("Gas Gap (Y=0..gap):", self.cfd_atmoheight_var, "mm"),
            ("Travel Window (Z):", self.cfd_travel_var, "mm"),
            ("Transverse Width (X, auto):", self.cfd_width_var, "mm"),
        ]
        for i, (label, var, unit) in enumerate(mesh_fields):
            lbl = tk.Label(lf_mesh, text=label, font=FONT_LABEL)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=(8, 4), pady=2)
            self._reg(lbl, "label")
            entry = tk.Entry(lf_mesh, textvariable=var, width=10, font=FONT_ENTRY, relief=tk.SOLID, bd=1)
            entry.grid(row=i, column=1, padx=4, pady=2, sticky=tk.W)
            self._reg(entry, "entry")
            u_lbl = tk.Label(lf_mesh, text=unit, font=FONT_SMALL)
            u_lbl.grid(row=i, column=2, sticky=tk.W, padx=4)
            self._reg(u_lbl, "label_dim")

        # --- MELTING Dict Overrides ---
        lf_melt = tk.LabelFrame(scroll_content, text="Melting Solver Controls",
                                font=FONT_SECTION, padx=8, pady=4)
        lf_melt.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_melt, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_melt, self._show_openfoam_info)

        self.cfd_mintcorr_var = tk.StringVar(value="1")
        self.cfd_maxtcorr_var = tk.StringVar(value="150")
        self.cfd_epstol_var = tk.StringVar(value="1e-6")
        self.cfd_epsrel_var = tk.StringVar(value="0.95")
        self.cfd_damper_var = tk.BooleanVar(value=False)

        melt_fields = [
            ("Min Temp Correctors:", self.cfd_mintcorr_var, "[-]"),
            ("Max Temp Correctors:", self.cfd_maxtcorr_var, "[-]"),
            ("Epsilon Tolerance:", self.cfd_epstol_var, "[-]"),
            ("Epsilon Relaxation:", self.cfd_epsrel_var, "[-]"),
        ]
        for i, (label, var, unit) in enumerate(melt_fields):
            lbl = tk.Label(lf_melt, text=label, font=FONT_LABEL)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=(8, 4), pady=2)
            self._reg(lbl, "label")
            entry = tk.Entry(lf_melt, textvariable=var, width=12, font=FONT_ENTRY, relief=tk.SOLID, bd=1)
            entry.grid(row=i, column=1, padx=4, pady=2, sticky=tk.W)
            self._reg(entry, "entry")
            u_lbl = tk.Label(lf_melt, text=unit, font=FONT_SMALL)
            u_lbl.grid(row=i, column=2, sticky=tk.W, padx=4)
            self._reg(u_lbl, "label_dim")

        cb_damp = tk.Checkbutton(lf_melt, text="Enable Interface Damper",
                                 variable=self.cfd_damper_var, font=FONT_LABEL)
        cb_damp.grid(row=len(melt_fields), column=0, columnspan=3, sticky=tk.W, padx=8, pady=2)
        self._reg(cb_damp, "check")

        # --- Beam Oscillation & Focus ---
        lf_beam = tk.LabelFrame(scroll_content, text="Beam Oscillation & Focus",
                                font=FONT_SECTION, padx=8, pady=4)
        lf_beam.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_beam, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_beam, self._show_openfoam_info)

        self.cfd_osc_amp_var = tk.StringVar(value="0.0")
        self.cfd_osc_freq_var = tk.StringVar(value="0.0")
        self.cfd_yr_var = tk.StringVar(value="0.15")
        self.cfd_focshift_var = tk.StringVar(value="0.0")

        beam_fields = [
            ("Oscillation Amplitude:", self.cfd_osc_amp_var, "mm"),
            ("Oscillation Frequency:", self.cfd_osc_freq_var, "Hz"),
            ("Y_R (Divergence Param):", self.cfd_yr_var, "[-]"),
            ("Focus Shift Velocity:", self.cfd_focshift_var, "mm/s"),
        ]
        for i, (label, var, unit) in enumerate(beam_fields):
            lbl = tk.Label(lf_beam, text=label, font=FONT_LABEL)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=(8, 4), pady=2)
            self._reg(lbl, "label")
            entry = tk.Entry(lf_beam, textvariable=var, width=12, font=FONT_ENTRY, relief=tk.SOLID, bd=1)
            entry.grid(row=i, column=1, padx=4, pady=2, sticky=tk.W)
            self._reg(entry, "entry")
            u_lbl = tk.Label(lf_beam, text=unit, font=FONT_SMALL)
            u_lbl.grid(row=i, column=2, sticky=tk.W, padx=4)
            self._reg(u_lbl, "label_dim")

        # --- Power Ramp ---
        lf_ramp = tk.LabelFrame(scroll_content, text="Power Ramp & Deposition Cutoff",
                                font=FONT_SECTION, padx=8, pady=4)
        lf_ramp.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_ramp, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_ramp, self._show_openfoam_info)

        self.cfd_qramp_var = tk.StringVar(value="auto")
        self.cfd_tshift_var = tk.StringVar(value="1.0")
        self.cfd_cutoff_var = tk.StringVar(value="0.5")

        ramp_fields = [
            ("Q Ramp Rate (auto=10x Q, decays past tshift):", self.cfd_qramp_var, "W/s"),
            ("Time Shift (tshift, keep > endTime):", self.cfd_tshift_var, "s"),
            ("Deposition Cutoff (alpha):", self.cfd_cutoff_var, "[-]"),
        ]
        for i, (label, var, unit) in enumerate(ramp_fields):
            lbl = tk.Label(lf_ramp, text=label, font=FONT_LABEL)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=(8, 4), pady=2)
            self._reg(lbl, "label")
            entry = tk.Entry(lf_ramp, textvariable=var, width=12, font=FONT_ENTRY, relief=tk.SOLID, bd=1)
            entry.grid(row=i, column=1, padx=4, pady=2, sticky=tk.W)
            self._reg(entry, "entry")
            u_lbl = tk.Label(lf_ramp, text=unit, font=FONT_SMALL)
            u_lbl.grid(row=i, column=2, sticky=tk.W, padx=4)
            self._reg(u_lbl, "label_dim")

        # --- Action Buttons ---
        btn_frame = tk.Frame(scroll_content, pady=6)
        btn_frame.pack(fill=tk.X, padx=8)
        self._reg(btn_frame, "card_frame")

        gen_btn = tk.Button(
            btn_frame, text="Generate BeamWeldFoam Case",
            font=FONT_STATUS,
            command=self._openfoam_generate_case, padx=12, pady=5
        )
        gen_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._reg(gen_btn, "card_frame")

        run_btn = tk.Button(
            btn_frame, text="Run BeamWeldFoam",
            font=FONT_STATUS,
            command=self._openfoam_run, padx=12, pady=5
        )
        run_btn.pack(side=tk.LEFT, padx=6)
        self._reg(run_btn, "card_frame")

        pv_btn = tk.Button(
            btn_frame, text="ParaView (OpenFOAM)",
            font=FONT_BTN,
            command=self._openfoam_paraview, padx=8, pady=4
        )
        pv_btn.pack(side=tk.LEFT, padx=6)
        self._reg(pv_btn, "card_frame")

        rep_btn = tk.Button(
            btn_frame, text="CFD Report",
            font=FONT_BTN,
            command=self._open_cfd_report, padx=8, pady=4
        )
        rep_btn.pack(side=tk.LEFT, padx=6)
        self._reg(rep_btn, "card_frame")

        # --- Status readout ---
        self.cfd_status_var = tk.StringVar(value="Ready. Configure parameters and click Generate.")
        cfd_status = tk.Label(scroll_content, textvariable=self.cfd_status_var,
                              font=FONT_MONO, anchor=tk.W, justify=tk.LEFT)
        cfd_status.pack(fill=tk.X, padx=12, pady=(4, 8))
        self._reg(cfd_status, "label_dim")

    def _collect_cfd_params(self) -> dict:
        """Collect CFD-specific parameters from the OpenFOAM tab.

        Never raises: unparsable entries fall back to port-validated defaults.
        'auto' (width / Q ramp) is passed through as a sentinel the generator
        resolves (bbox-derived width, 10xQ ramp).
        """
        def _f(var, default=0.0):
            try:
                txt = var.get().strip()
            except (ValueError, AttributeError):
                return default
            if txt.lower() in ("", "auto", "none", "default"):
                return default
            try:
                return float(txt)
            except ValueError:
                return default

        def _raw(var, default="auto"):
            try:
                txt = var.get().strip()
                return txt if txt else default
            except (ValueError, AttributeError):
                return default

        cfd = {}
        cfd["endTime"] = _f(self.cfd_endtime_var, 5.0e-4)
        cfd["deltaT"] = _f(self.cfd_deltat_var, 1e-7)
        cfd["writeInterval"] = _f(self.cfd_writeint_var, 5.0e-5)
        cfd["maxCo"] = _f(self.cfd_maxco_var, 0.2)
        cfd["maxAlphaCo"] = _f(self.cfd_maxalphaco_var, 0.2)
        cfd["maxDeltaT"] = _f(self.cfd_maxdeltat_var, 1e-5)
        cfd["cell_size_mm"] = _f(self.cfd_cellsize_var, 1.0)
        cfd["atmosphere_height"] = _f(self.cfd_atmoheight_var, 3.0) / 1000.0  # mm -> m
        cfd["travel_length_mm"] = _raw(self.cfd_travel_var, "40.0")
        cfd["domain_width_mm"] = _raw(self.cfd_width_var, "auto")
        cfd["minTempCorrector"] = int(_f(self.cfd_mintcorr_var, 1))
        cfd["maxTempCorrector"] = int(_f(self.cfd_maxtcorr_var, 150))
        cfd["epsilonTolerance"] = _f(self.cfd_epstol_var, 1e-6)
        cfd["epsilonRelaxation"] = _f(self.cfd_epsrel_var, 0.95)
        cfd["damperSwitch"] = self.cfd_damper_var.get()
        cfd["oscillation_amplitude"] = _f(self.cfd_osc_amp_var, 0.0) / 1000.0  # mm -> m
        cfd["oscillation_frequency"] = _f(self.cfd_osc_freq_var, 0.0)
        cfd["Y_R"] = _f(self.cfd_yr_var, 0.15)
        cfd["foc_shift_vel"] = _f(self.cfd_focshift_var, 0.0) / 1000.0  # mm/s -> m/s
        cfd["tshift"] = _f(self.cfd_tshift_var, 1.0)
        cfd["deposition_cutoff"] = _f(self.cfd_cutoff_var, 0.5)

        # Q ramp rate: auto (or garbage) means 10x arc power (generator resolves it)
        qr_str = self.cfd_qramp_var.get().strip().lower()
        if qr_str == "auto":
            cfd["Q_ramp_rate"] = None  # let the generator compute it
        else:
            try:
                cfd["Q_ramp_rate"] = float(qr_str)
            except ValueError:
                cfd["Q_ramp_rate"] = None

        return cfd

    def _cfd_case_subdir(self, run_dir: str) -> str:
        """CFD case lives in <run_dir>/openfoam/ next to the EDP script."""
        return os.path.join(run_dir, "openfoam")

    def _find_cfd_case_dir(self, run_dir: str):
        """Locate the CFD case, accepting the legacy openfoam_meltpool name."""
        for name in ("openfoam", "openfoam_meltpool"):
            cand = os.path.join(run_dir, name)
            if os.path.exists(cand):
                return cand
        return None

    def _openfoam_generate_case(self):
        """Generate a beamWeldFoam case under <run_dir>/openfoam/ (FreeFEM-style)."""
        try:
            run_dir = self._get_or_create_run_dir(is_new_run=False)
            foam_dir = self._cfd_case_subdir(run_dir)
            params = self._collect_params()
            cfd_params = self._collect_cfd_params()

            from core.fluid.openfoam_coupling import setup_beamweldfoam_case
            result = setup_beamweldfoam_case(foam_dir, params, cfd_params)

            def _num(x, fb=0.0):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return fb

            Q_arc = _num(params.get("I_weld", 200)) * _num(params.get("U_weld", 24)) * _num(params.get("eta_arc", 0.8))
            v_mms = _num(params.get("v_scan", 0.01)) * 1000.0
            self._log(f"\n{'='*55}\n")
            self._log(f"[OpenFOAM 2412] beamWeldFoam case generated\n")
            self._log(f"  Case dir:  {result}\n")
            self._log(f"  Material:  {params.get('material_dep', 'ER70S-6')}\n")
            self._log(f"  Arc Power: {Q_arc:.1f} W\n")
            self._log(f"  Scan Speed: {v_mms:.1f} mm/s\n")
            self._log(f"  End Time:  {_num(cfd_params.get('endTime', 5.0e-4)):.6g} s\n")
            self._log(f"  Cell Size: {_num(cfd_params.get('cell_size_mm', 1.0)):.2f} mm\n")
            self._log(f"{'='*55}\n")

            self.cfd_status_var.set(f"Case generated at: {foam_dir}")
            self._cfd_case_dir = foam_dir
        except Exception as e:
            self._log(f"\n[OpenFOAM] [ERROR] Case generation failed: {e}\n")
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror("OpenFOAM Case Error", str(e))

    def _openfoam_run(self):
        """Run the beamWeldFoam solver on the generated case, streaming output."""
        foam_dir = getattr(self, "_cfd_case_dir", None)
        if not foam_dir or not os.path.exists(foam_dir):
            # Try to generate first
            self._openfoam_generate_case()
            foam_dir = getattr(self, "_cfd_case_dir", None)
            if not foam_dir:
                found = self._find_cfd_case_dir(self._get_active_or_latest_run_dir())
                if found:
                    foam_dir = found
                    self._cfd_case_dir = found
                else:
                    return

        allrun = os.path.join(foam_dir, "Allrun")
        if not os.path.exists(allrun):
            self._log(f"[OpenFOAM] [ERROR] Allrun script not found: {allrun}\n")
            return

        self._log(f"\n{'='*55}\nSTARTING beamWeldFoam SOLVER...\n{'='*55}\n")
        self.cfd_status_var.set("Running beamWeldFoam...")
        self.status_var.set("Status: beamWeldFoam Running...")

        def run_foam_thread():
            try:
                log_path = os.path.join(foam_dir, "sim_openfoam.log")
                # Portable: run Allrun directly (it discovers the OpenFOAM env itself).
                # Same pseudo-terminal trick as FreeFEM to force live output.
                _use_script = shutil.which("script") is not None
                if _use_script:
                    cmd = ["script", "-q", "/dev/null", "bash", allrun]
                else:
                    cmd = ["bash", allrun]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=foam_dir,
                    bufsize=1
                )

                import time as _time
                with open(log_path, "w", encoding="utf-8") as log_f:
                    log_f.write(f"beamWeldFoam Solver Log\nCase: {foam_dir}\n\n")
                    # A solver prints thousands of lines; forwarding EVERY one
                    # through root.after() queued tens of thousands of Tk events
                    # and froze the UI on long runs. The file still gets every
                    # line; the widget gets a throttled digest (~4 Hz) plus the
                    # progress lines.
                    _pending: list = []
                    _last_flush = 0.0

                    def _flush(force=False):
                        nonlocal _last_flush
                        now = _time.monotonic()
                        if _pending and (force or now - _last_flush >= 0.25):
                            chunk = "".join(_pending)
                            del _pending[:]
                            _last_flush = now
                            self.root.after(0, self._log, chunk)

                    for line in proc.stdout:
                        line = line.replace("\r", "")
                        while line and ord(line[0]) < 32:
                            line = line[1:]
                        log_f.write(line)
                        log_f.flush()
                        if "Time =" in line or "Courant" in line or "ExecutionTime" in line:
                            self.root.after(0, self._log, line)
                        else:
                            _pending.append(line)
                            if len(_pending) > 400:
                                _flush(force=True)
                            else:
                                _flush()
                    _flush(force=True)

                    # Track progress from the solver log tail (the live "Time ="
                    # lines were already forwarded above; re-scan the file so a
                    # status line exists even if the widget digest skipped it).
                    try:
                        with open(log_path, "r", encoding="utf-8",
                                  errors="replace") as _lf:
                            _tail = [ln.strip() for ln in _lf if "Time =" in ln]
                        if _tail:
                            self.root.after(0, lambda m=_tail[-1]: (
                                self.cfd_status_var.set(f"Solving: {m}")))
                    except Exception:
                        pass

                proc.wait()
                rc = proc.returncode
                if rc == 0:
                    self.root.after(0, self._log, f"\n[OpenFOAM] [OK] beamWeldFoam completed successfully.\n")
                    self.root.after(0, self.status_var.set, "Status: beamWeldFoam Complete")
                    self.root.after(0, self.cfd_status_var.set, "Solver completed. Postprocessing...")
                    # Auto-run the case-local CFD postprocessor (log + field history).
                    try:
                        post = subprocess.run(
                            [sys.executable, os.path.join(foam_dir, "postprocess_cfd.py"), foam_dir],
                            capture_output=True, text=True, cwd=foam_dir, timeout=180)
                        out = (post.stdout or "") + (post.stderr or "")
                        tail = "\n".join(out.strip().splitlines()[-25:])
                        self.root.after(0, self._log, f"\n[CFD-POST] postprocess_cfd.py finished:\n{tail}\n")
                    except Exception as pe:
                        self.root.after(0, self._log, f"\n[CFD-POST] [WARNING] Postprocess note: {pe}\n")
                    # Auto-build the CFD HTML dossier and publish it to the case folder.
                    try:
                        from core.postprocess.output_manager import generate_cfd_html_report
                        cfd_rep = generate_cfd_html_report(foam_dir)
                        case_top = os.path.dirname(os.path.abspath(foam_dir))
                        import shutil as _sh2
                        top_cfd = os.path.join(case_top, "cfd_report.html")
                        _sh2.copy2(cfd_rep, top_cfd)
                        self.root.after(0, self._log, f"[CFD-REPORT] HTML dossier ready: {cfd_rep}\n")
                        self.root.after(0, self._log, f"[CFD-REPORT] Published to case folder: {top_cfd}\n")
                    except Exception as re_:
                        self.root.after(0, self._log, f"\n[CFD-REPORT] [WARNING] Report note: {re_}\n")
                    self.root.after(0, self.cfd_status_var.set, "Solver completed. Open results in ParaView.")
                else:
                    self.root.after(0, self._log, f"\n[OpenFOAM] [ERROR] beamWeldFoam exited with code {rc}\n")
                    self.root.after(0, self.status_var.set, f"Status: beamWeldFoam Error (Exit {rc})")
                    self.root.after(0, self.cfd_status_var.set, f"Error: solver exit code {rc}. Check log.")
            except Exception as e:
                self.root.after(0, self._log, f"\n[OpenFOAM] [ERROR] Launch failed: {e}\n")
                self.root.after(0, self.status_var.set, "Status: beamWeldFoam Error")

        threading.Thread(target=run_foam_thread, daemon=True).start()

    def _open_cfd_report(self):
        """Open the CFD HTML dossier (generating it on the fly if results exist)."""
        import webbrowser
        foam_dir = getattr(self, "_cfd_case_dir", None)
        if not foam_dir or not os.path.exists(foam_dir):
            foam_dir = self._find_cfd_case_dir(self._get_active_or_latest_run_dir())
        if not foam_dir or not os.path.exists(foam_dir):
            messagebox.showinfo("No CFD Case", "No OpenFOAM case directory found.\nGenerate and run a case first.")
            return
        rep = os.path.join(foam_dir, "cfd_report.html")
        hist = os.path.join(foam_dir, "cfd_history.csv")
        stale = (not os.path.exists(rep) or
                 (os.path.exists(hist) and os.path.getmtime(hist) > os.path.getmtime(rep)))
        if stale:
            try:
                from core.postprocess.output_manager import generate_cfd_html_report
                rep = generate_cfd_html_report(foam_dir)
                import shutil as _sh3
                _sh3.copy2(rep, os.path.join(os.path.dirname(os.path.abspath(foam_dir)), "cfd_report.html"))
            except Exception as e:
                messagebox.showinfo("CFD Report", f"Could not generate report: {e}")
                return
        webbrowser.open(f"file://{os.path.abspath(rep)}")
        self._log(f"[CFD-REPORT] Opened CFD dossier: {os.path.abspath(rep)}\n")

    def _openfoam_paraview(self):
        """Open the OpenFOAM case results in ParaView."""
        foam_dir = getattr(self, "_cfd_case_dir", None)
        if not foam_dir or not os.path.isdir(str(foam_dir)):
            resolved = self._find_cfd_case_dir(self._get_active_or_latest_run_dir())
            foam_dir = str(resolved) if resolved else None

        if not foam_dir or not os.path.isdir(foam_dir):
            messagebox.showinfo("No CFD Case", "No OpenFOAM case directory found.\nGenerate and run a case first.")
            return

        # Look for time directories with results
        foam_file = os.path.join(foam_dir, "case.foam")
        if not os.path.exists(foam_file):
            with open(foam_file, "w") as f:
                f.write("")  # empty .foam file for ParaView OpenFOAM reader

        # Locate ParaView binary (portable: PATH first, then well-known spots, no hardcoded home dir)
        candidates = [
            shutil.which("paraview"),
            os.environ.get("PARAVIEW_BIN"),
            os.path.expanduser("~/bin/paraview"),
            "/Applications/ParaView-5.13.3.app/Contents/MacOS/paraview",
            "/opt/homebrew/bin/paraview",
        ] + glob.glob("/Applications/ParaView*.app/Contents/MacOS/paraview")

        pv_bin = None
        for c in candidates:
            if c and os.path.exists(c):
                pv_bin = c
                break

        if pv_bin:
            try:
                subprocess.Popen([pv_bin, foam_file])
                self._log(f"[PARAVIEW] [OK] Opening OpenFOAM case: {foam_file}\n")
            except Exception as e:
                self._log(f"[PARAVIEW] [ERROR] {e}\n")
        elif sys.platform == "darwin":
            pv_apps = glob.glob("/Applications/ParaView*.app")
            if pv_apps:
                subprocess.Popen(["open", "-a", pv_apps[0], foam_file])
                self._log(f"[PARAVIEW] [OK] Opening via {pv_apps[0]}\n")
            else:
                subprocess.Popen(["open", foam_dir])
                self._log(f"[PARAVIEW] ParaView not found. Opened case directory.\n")
        else:
            subprocess.Popen(["open", foam_dir])

    def _build_parametric_tab(self):
        """Dedicated Parametric Sweep & Design of Experiments (DOE) Tab."""
        self.parametric_tab = tk.Frame(self.notebook)
        self.notebook.add(self.parametric_tab, text="Parametric & DOE")
        self._reg(self.parametric_tab, "card_frame")

        content = tk.Frame(self.parametric_tab, padx=12, pady=10)
        content.pack(fill=tk.BOTH, expand=True)
        self._reg(content, "card_frame")

        hdr = tk.Label(
            content,
            text="Parametric Design of Experiments (DOE) & Multi-Factor Sensitivity",
            font=FONT_TITLE, anchor=tk.W
        )
        hdr.pack(fill=tk.X, pady=(0, 2))
        self._reg(hdr, "label")

        sub = tk.Label(
            content,
            text="Automated multi-run parameter sweep engine: dynamically vary heat input, ultrasonic vibration, and layer counts.",
            font=FONT_SMALL, fg="#64748b", anchor=tk.W
        )
        sub.pack(fill=tk.X, pady=(0, 8))
        self._reg(sub, "label")

        split_pane = tk.PanedWindow(content, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        split_pane.pack(fill=tk.BOTH, expand=True)

        left_ctrl = tk.Frame(split_pane, padx=8, pady=6)
        split_pane.add(left_ctrl, width=420, minsize=360)
        self._reg(left_ctrl, "card_frame")

        # 1. Heat Input Variation Box
        hi_box = tk.LabelFrame(left_ctrl, text="1. Heat Input Variation", font=FONT_SECTION, padx=8, pady=6)
        hi_box.pack(fill=tk.X, pady=(0, 8))
        self._reg(hi_box, "card_labelframe")

        self.sweep_i_var = tk.BooleanVar(value=True)
        tk.Checkbutton(hi_box, text="Vary Current I (A):", variable=self.sweep_i_var, font=FONT_SMALL_BOLD).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.sweep_i_entry = ttk.Entry(hi_box, width=20)
        self.sweep_i_entry.insert(0, "150.0, 180.0, 210.0")
        self.sweep_i_entry.grid(row=0, column=1, sticky=tk.W, padx=6, pady=2)

        self.sweep_v_var = tk.BooleanVar(value=False)
        tk.Checkbutton(hi_box, text="Vary Speed v (mm/s):", variable=self.sweep_v_var, font=FONT_SMALL_BOLD).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.sweep_v_entry = ttk.Entry(hi_box, width=20)
        self.sweep_v_entry.insert(0, "6.0, 8.0, 10.0")
        self.sweep_v_entry.grid(row=1, column=1, sticky=tk.W, padx=6, pady=2)

        # 2. Ultrasonic Variation Box
        us_box = tk.LabelFrame(left_ctrl, text="2. Ultrasonic Vibration Variation", font=FONT_SECTION, padx=8, pady=6)
        us_box.pack(fill=tk.X, pady=(0, 8))
        self._reg(us_box, "card_labelframe")

        self.sweep_us_var = tk.BooleanVar(value=True)
        tk.Checkbutton(us_box, text="Vary Freq f_US (kHz):", variable=self.sweep_us_var, font=FONT_SMALL_BOLD).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.sweep_us_entry = ttk.Entry(us_box, width=20)
        self.sweep_us_entry.insert(0, "0.0, 20.0, 40.0")
        self.sweep_us_entry.grid(row=0, column=1, sticky=tk.W, padx=6, pady=2)

        # 3. Layer Count Variation Box
        lay_box = tk.LabelFrame(left_ctrl, text="3. Layer Count Variation", font=FONT_SECTION, padx=8, pady=6)
        lay_box.pack(fill=tk.X, pady=(0, 8))
        self._reg(lay_box, "card_labelframe")

        self.sweep_lay_var = tk.BooleanVar(value=False)
        tk.Checkbutton(lay_box, text="Vary Layers N_layers:", variable=self.sweep_lay_var, font=FONT_SMALL_BOLD).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.sweep_lay_entry = ttk.Entry(lay_box, width=20)
        self.sweep_lay_entry.insert(0, "5, 10, 20")
        self.sweep_lay_entry.grid(row=0, column=1, sticky=tk.W, padx=6, pady=2)

        # 4. Wire Feedstock Variation Box
        wire_box = tk.LabelFrame(left_ctrl, text="4. Wire Feedstock Variation", font=FONT_SECTION, padx=8, pady=6)
        wire_box.pack(fill=tk.X, pady=(0, 8))
        self._reg(wire_box, "card_labelframe")

        self.sweep_dwire_var = tk.BooleanVar(value=False)
        tk.Checkbutton(wire_box, text="Vary Wire Dia d_wire (mm):", variable=self.sweep_dwire_var, font=FONT_SMALL_BOLD).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.sweep_dwire_entry = ttk.Entry(wire_box, width=20)
        self.sweep_dwire_entry.insert(0, "0.8, 1.2, 1.6")
        self.sweep_dwire_entry.grid(row=0, column=1, sticky=tk.W, padx=6, pady=2)

        # Action Buttons
        btn_box = tk.Frame(left_ctrl, pady=6)
        btn_box.pack(fill=tk.X)
        self._reg(btn_box, "card_frame")

        gen_doe_btn = tk.Button(
            btn_box, text="⚡ Generate Matrix", font=FONT_BTN,
            bg=self.colors["button"], fg=self.colors["button_fg"],
            command=self._generate_doe_matrix, padx=8, pady=4
        )
        gen_doe_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.run_sweep_btn = tk.Button(
            btn_box, text="▶ Run Batch Sweep", font=FONT_STATUS,
            bg=self.colors["button_primary"], fg=self.colors["button_primary_fg"],
            activebackground=self.colors["button_primary_active"],
            command=self._run_parametric_sweep, padx=10, pady=4
        )
        self.run_sweep_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.sweep_status_lbl = tk.Label(left_ctrl, text="Queue: 0 runs pending", font=FONT_SMALL, fg="#64748b", anchor=tk.W)
        self.sweep_status_lbl.pack(fill=tk.X, pady=4)
        self._reg(self.sweep_status_lbl, "label")

        # Right Panel: Matrix Queue Treeview
        right_view = tk.Frame(split_pane, padx=8, pady=6)
        split_pane.add(right_view, width=540, minsize=400)
        self._reg(right_view, "card_frame")

        q_lbl = tk.Label(right_view, text="Parametric Run Matrix & Execution Status", font=FONT_SECTION, anchor=tk.W)
        q_lbl.pack(fill=tk.X, pady=(0, 4))
        self._reg(q_lbl, "label")

        cols = ("run_id", "current", "speed", "us_freq", "layers", "wire_dia", "status", "tpeak")
        self.doe_tree = ttk.Treeview(right_view, columns=cols, show="headings", height=14)
        self.doe_tree.heading("run_id", text="Run ID")
        self.doe_tree.heading("current", text="Current I (A)")
        self.doe_tree.heading("speed", text="Speed (mm/s)")
        self.doe_tree.heading("us_freq", text="f_US (kHz)")
        self.doe_tree.heading("layers", text="Layers")
        self.doe_tree.heading("wire_dia", text="Wire ∅ (mm)")
        self.doe_tree.heading("status", text="Status")
        self.doe_tree.heading("tpeak", text="T_peak (°C)")

        self.doe_tree.column("run_id", width=75, anchor=tk.CENTER)
        self.doe_tree.column("current", width=80, anchor=tk.CENTER)
        self.doe_tree.column("speed", width=80, anchor=tk.CENTER)
        self.doe_tree.column("us_freq", width=75, anchor=tk.CENTER)
        self.doe_tree.column("layers", width=60, anchor=tk.CENTER)
        self.doe_tree.column("wire_dia", width=75, anchor=tk.CENTER)
        self.doe_tree.column("status", width=85, anchor=tk.CENTER)
        self.doe_tree.column("tpeak", width=75, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(right_view, orient=tk.VERTICAL, command=self.doe_tree.yview)
        self.doe_tree.configure(yscrollcommand=tree_scroll.set)
        self.doe_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._doe_queue = []

    def _parse_csv_num_list(self, text: str, default_val: float) -> list:
        try:
            items = [float(x.strip()) for x in text.split(",") if x.strip()]
            return items if items else [default_val]
        except Exception:
            return [default_val]

    def _generate_doe_matrix(self):
        """Build DOE Cartesian product of all selected sweep dimensions."""
        for row in self.doe_tree.get_children():
            self.doe_tree.delete(row)
        self._doe_queue.clear()

        cur_str = self._param_vars.get("I_weld", (tk.StringVar(value="150.0"), "float"))[0].get() or "150.0"
        spd_str = self._param_vars.get("v_scan", (tk.StringVar(value="10.0"), "float"))[0].get() or "10.0"
        lay_str = self._param_vars.get("n_layers", (tk.StringVar(value="10"), "int"))[0].get() or "10"
        dw_str = self._param_vars.get("d_wire", (tk.StringVar(value="1.2"), "float"))[0].get() or "1.2"

        currents = self._parse_csv_num_list(self.sweep_i_entry.get(), 150.0) if self.sweep_i_var.get() else [float(cur_str)]
        speeds = self._parse_csv_num_list(self.sweep_v_entry.get(), 10.0) if self.sweep_v_var.get() else [float(spd_str)]
        us_freqs = self._parse_csv_num_list(self.sweep_us_entry.get(), 0.0) if self.sweep_us_var.get() else [0.0]
        layers = [int(x) for x in self._parse_csv_num_list(self.sweep_lay_entry.get(), 10.0)] if self.sweep_lay_var.get() else [int(lay_str)]
        dwires = self._parse_csv_num_list(self.sweep_dwire_entry.get(), 1.2) if self.sweep_dwire_var.get() else [float(dw_str)]

        import itertools
        run_idx = 1
        for cur, spd, us, lay, dw in itertools.product(currents, speeds, us_freqs, layers, dwires):
            rid = f"DOE_{run_idx:02d}"
            item = {
                "run_id": rid,
                "current": cur,
                "speed": spd,
                "us_freq": us,
                "layers": lay,
                "wire_dia": dw,
                "status": "Ready",
                "tpeak": "-"
            }
            self._doe_queue.append(item)
            self.doe_tree.insert("", tk.END, iid=rid, values=(rid, f"{cur:.1f}", f"{spd:.1f}", f"{us:.1f}", lay, f"{dw:.2f}", "Ready", "-"))
            run_idx += 1

        self.sweep_status_lbl.config(text=f"Queue: {len(self._doe_queue)} runs generated")

    def _run_parametric_sweep(self):
        """Execute all parameter configurations in the DOE queue sequentially through EDPBuilder."""
        if not self._doe_queue:
            self._generate_doe_matrix()
        if not self._doe_queue:
            return

        import threading
        def worker():
            for item in self._doe_queue:
                rid = item["run_id"]
                self.doe_tree.set(rid, "status", "Running...")
                self.sweep_status_lbl.config(text=f"Executing {rid} ({item['current']}A, ∅{item['wire_dia']}mm, {item['layers']}L)...")

                p = self._collect_params()
                p["I_weld"] = float(item["current"])
                p["v_scan"] = float(item["speed"]) / 1000.0 if float(item["speed"]) > 0.05 else float(item["speed"])
                p["ultrasonic_frequency"] = float(item["us_freq"]) * 1000.0 if float(item["us_freq"]) > 0 else 0.0
                p["n_layers"] = int(item["layers"])
                p["num_layers"] = int(item["layers"])
                p["d_wire"] = float(item["wire_dia"]) / 1000.0 if float(item["wire_dia"]) > 0.05 else float(item["wire_dia"])
                p["wire_diameter"] = p["d_wire"]
                p["wire_diameter_mm"] = p["d_wire"] * 1000.0

                builder = EDPBuilder(p)
                run_dir = self._get_or_create_run_dir(is_new_run=True)
                fem_dir = self._fem_subdir(run_dir)
                edp_path = os.path.join(fem_dir, "waam_simulation.edp")
                builder.save(edp_path)

                ff_path = shutil.which("FreeFem++") or "/Applications/FreeFem++.app/Contents/ff-4.15/bin/FreeFem++"
                sim_log = os.path.join(fem_dir, "sim.log")
                with open(sim_log, "w") as lf:
                    proc = subprocess.Popen([ff_path, edp_path, fem_dir + "/"], stdout=lf, stderr=subprocess.STDOUT)
                    proc.wait()

                t_peak = "-"
                try:
                    with open(sim_log, "r") as lf:
                        content = lf.read()
                        matches = re.findall(r"Tmax=([\d\.]+)C", content)
                        if matches:
                            t_peak = f"{float(matches[-1]):.1f}"
                except Exception:
                    pass

                self.doe_tree.set(rid, "status", "Done")
                self.doe_tree.set(rid, "tpeak", t_peak)

            self.sweep_status_lbl.config(text="Parametric sweep completed!")
            messagebox.showinfo("Sweep Complete", f"Completed all {len(self._doe_queue)} parametric runs successfully.")

        threading.Thread(target=worker, daemon=True).start()

    def _build_viewport_tab(self):
        """Dedicated full-size 3D Finite Element Mesh & Geometry Viewport Tab."""
        self.viewport_tab = tk.Frame(self.notebook)
        self.notebook.add(self.viewport_tab, text="3D Viewport")
        self._reg(self.viewport_tab, "card_frame")

        from gui.widgets.mesh_viewer_3d import EmbeddedMeshViewport
        self.viewport = EmbeddedMeshViewport(self.viewport_tab, app=self)
        self.viewport.pack(fill=tk.BOTH, expand=True)
        self.viewport_3d = self.viewport

    def _build_default_case_tab(self):
        """Dedicated tab to view the live FreeFEM++ .edp script generated dynamically by EDPBuilder."""
        self.default_case_tab = tk.Frame(self.notebook)
        self.notebook.add(self.default_case_tab, text="Generated Script Preview")
        self._reg(self.default_case_tab, "card_frame")

        ctrl_bar = tk.Frame(self.default_case_tab, padx=6, pady=4)
        ctrl_bar.pack(fill=tk.X)
        self._reg(ctrl_bar, "card_frame")

        ref_btn = tk.Button(
            ctrl_bar, text="🔄 Refresh Script Preview", font=FONT_BTN,
            bg=self.colors["button_primary"], fg=self.colors["button_primary_fg"],
            activebackground=self.colors["button_primary_active"],
            command=self._refresh_generated_edp_preview, padx=8, pady=3
        )
        ref_btn.pack(side=tk.LEFT, padx=(0, 8))

        lbl_info = tk.Label(
            ctrl_bar, text="ⓘ Live C++/FreeFEM++ simulation script synthesized dynamically from active parameters (Zero hardcoding).",
            font=FONT_SMALL, fg="#0284c7"
        )
        lbl_info.pack(side=tk.LEFT)
        self._reg(lbl_info, "label")

        self.default_case_text = scrolledtext.ScrolledText(
            self.default_case_tab,
            font=FONT_MONO_LG, wrap=tk.NONE
        )
        self.default_case_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._refresh_generated_edp_preview()

    def _refresh_generated_edp_preview(self):
        """Dynamically render the exact FreeFEM++ .edp script that EDPBuilder generates from current inputs."""
        try:
            params = self._collect_params()
            builder = EDPBuilder(params)
            code = builder.build()
            self.default_case_text.config(state=tk.NORMAL)
            self.default_case_text.delete("1.0", tk.END)
            self.default_case_text.insert(tk.END, code)
            self.default_case_text.config(state=tk.DISABLED)
        except Exception as e:
            self.default_case_text.config(state=tk.NORMAL)
            self.default_case_text.delete("1.0", tk.END)
            self.default_case_text.insert(tk.END, f"// Error generating script preview from parameters:\n// {e}")
            self.default_case_text.config(state=tk.DISABLED)

    
    def _build_claude_tab(self):
        """Dedicated tab for Ultrasonic and Multi-Material WAAM Solver."""
        self.claude_tab = tk.Frame(self.notebook)
        self.notebook.add(self.claude_tab, text="Ultrasonic")
        self._reg(self.claude_tab, "card_frame")

        content = tk.Frame(self.claude_tab, padx=14, pady=10)
        content.pack(fill=tk.BOTH, expand=True)
        self._reg(content, "card_frame")

        title = tk.Label(
            content, text="Ultrasonic & Multi-Material WAAM Solver",
            font=FONT_TITLE, anchor=tk.W
        )
        title.pack(fill=tk.X, pady=(0, 4))
        self._reg(title, "label")

        desc = tk.Label(
            content,
            text="High-fidelity 3D multi-material thermo-mechanical additive simulation with Ultrasonic Acoustic Softening\n"
                 "and exact conforming mesh (zero hanging nodes).",
            font=FONT_SUBTITLE, justify=tk.LEFT, anchor=tk.W
        )
        desc.pack(fill=tk.X, pady=(0, 6))
        self._reg(desc, "label")

        # Physics Module Gating Status Card
        status_card = tk.Frame(content, padx=8, pady=6, bd=1, relief=tk.SOLID)
        status_card.pack(fill=tk.X, pady=(0, 8))
        self._reg(status_card, "card_frame")

        self.us_physics_status_lbl = tk.Label(
            status_card,
            text="⚠ Ultrasonic Physics Module: DISABLED (Physics Tab OFF — Zero Acoustic Softening Applied)",
            font=FONT_SMALL_BOLD, fg="#ea580c", anchor=tk.W
        )
        self.us_physics_status_lbl.pack(side=tk.LEFT, padx=4)
        self._reg(self.us_physics_status_lbl, "label")

        self.us_toggle_btn = tk.Button(
            status_card, text="Enable in Physics Tab", font=FONT_SMALL_BOLD,
            relief=tk.FLAT, padx=8, pady=2, cursor="hand2",
            command=self._toggle_ultrasonic_from_tab
        )
        self.us_toggle_btn.pack(side=tk.RIGHT, padx=4)
        self._reg(self.us_toggle_btn, "button")

        # --- Ultrasonic & Geometry Parameters ---
        param_lf = tk.LabelFrame(content, text="Simulation Parameters", font=FONT_SECTION, padx=10, pady=8)
        param_lf.pack(fill=tk.X, pady=(0, 10))
        self._reg(param_lf, "card_labelframe")
        self._add_info_btn_to_labelframe(param_lf, self._show_ultrasonic_info)

        grid = tk.Frame(param_lf)
        grid.pack(fill=tk.X)
        self._reg(grid, "card_frame")

        # Row 0
        tk.Label(grid, text="Ultrasonic Freq (Hz):", font=FONT_SMALL_BOLD).grid(row=0, column=0, sticky=tk.W, pady=3)
        self.us_freq_var = tk.StringVar(value="20000")
        ttk.Entry(grid, textvariable=self.us_freq_var, width=12).grid(row=0, column=1, sticky=tk.W, padx=6, pady=3)

        tk.Label(grid, text="Total Layers:", font=FONT_SMALL_BOLD).grid(row=0, column=2, sticky=tk.W, padx=(15, 0), pady=3)
        self.us_layers_var = tk.StringVar(value="10")
        ttk.Entry(grid, textvariable=self.us_layers_var, width=12).grid(row=0, column=3, sticky=tk.W, padx=6, pady=3)

        # Row 1
        tk.Label(grid, text="Bead Width (mm):", font=FONT_SMALL_BOLD).grid(row=1, column=0, sticky=tk.W, pady=3)
        self.us_width_var = tk.StringVar(value="5.0")
        ttk.Entry(grid, textvariable=self.us_width_var, width=12).grid(row=1, column=1, sticky=tk.W, padx=6, pady=3)

        tk.Label(grid, text="First Layer Height (mm):", font=FONT_SMALL_BOLD).grid(row=1, column=2, sticky=tk.W, padx=(15, 0), pady=3)
        self.us_height_var = tk.StringVar(value="4.0")
        ttk.Entry(grid, textvariable=self.us_height_var, width=12).grid(row=1, column=3, sticky=tk.W, padx=6, pady=3)

        # Row 2: Remelt / Fusion Ratio and Dynamic Subsequent Height Info
        tk.Label(grid, text="Remelt Fusion (%):", font=FONT_SMALL_BOLD).grid(row=2, column=0, sticky=tk.W, pady=3)
        self.us_remelt_var = tk.StringVar(value="80")
        ttk.Entry(grid, textvariable=self.us_remelt_var, width=12).grid(row=2, column=1, sticky=tk.W, padx=6, pady=3)

        self.us_fusion_info_lbl = tk.Label(grid, text="Subsequent: 3.20 mm (80% fusion)", font=FONT_SMALL_BOLD, fg="#2563eb")
        self.us_fusion_info_lbl.grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=(15, 0), pady=3)
        self._reg(self.us_fusion_info_lbl, "label")

        # Row 3: UVA Physics Parameters
        tk.Label(grid, text="UVA Amplitude (um):", font=FONT_SMALL_BOLD).grid(row=3, column=0, sticky=tk.W, pady=3)
        self.us_amp_var = tk.StringVar(value="15.0")
        ttk.Entry(grid, textvariable=self.us_amp_var, width=12).grid(row=3, column=1, sticky=tk.W, padx=6, pady=3)

        tk.Label(grid, text="Coupling Eff (0-1):", font=FONT_SMALL_BOLD).grid(row=3, column=2, sticky=tk.W, padx=(15, 0), pady=3)
        self.us_coupling_var = tk.StringVar(value="0.15")
        ttk.Entry(grid, textvariable=self.us_coupling_var, width=12).grid(row=3, column=3, sticky=tk.W, padx=6, pady=3)

        # Row 4: Attenuation and Sound Speed
        tk.Label(grid, text="Attenuation (1/m):", font=FONT_SMALL_BOLD).grid(row=4, column=0, sticky=tk.W, pady=3)
        self.us_attenuation_var = tk.StringVar(value="0.12")
        ttk.Entry(grid, textvariable=self.us_attenuation_var, width=12).grid(row=4, column=1, sticky=tk.W, padx=6, pady=3)

        tk.Label(grid, text="Sound Speed (m/s):", font=FONT_SMALL_BOLD).grid(row=4, column=2, sticky=tk.W, padx=(15, 0), pady=3)
        self.us_csound_var = tk.StringVar(value="5790")
        ttk.Entry(grid, textvariable=self.us_csound_var, width=12).grid(row=4, column=3, sticky=tk.W, padx=6, pady=3)

        # Row 5: Streaming k and Cavitation threshold
        tk.Label(grid, text="Streaming k (W/m/K):", font=FONT_SMALL_BOLD).grid(row=5, column=0, sticky=tk.W, pady=3)
        self.us_kstream_var = tk.StringVar(value="45.0")
        ttk.Entry(grid, textvariable=self.us_kstream_var, width=12).grid(row=5, column=1, sticky=tk.W, padx=6, pady=3)

        tk.Label(grid, text="Cavitation thresh (MPa):", font=FONT_SMALL_BOLD).grid(row=5, column=2, sticky=tk.W, padx=(15, 0), pady=3)
        self.us_cavthresh_var = tk.StringVar(value="1.0")
        ttk.Entry(grid, textvariable=self.us_cavthresh_var, width=12).grid(row=5, column=3, sticky=tk.W, padx=6, pady=3)

        # --- UVA Physics Modules ---
        uva_lf = tk.LabelFrame(content, text="UVA Physics Modules", font=FONT_SECTION, padx=10, pady=8)
        uva_lf.pack(fill=tk.X, pady=(0, 10))
        self._reg(uva_lf, "card_labelframe")
        self._add_info_btn_to_labelframe(uva_lf, self._show_ultrasonic_info)

        uva_grid = tk.Frame(uva_lf)
        uva_grid.pack(fill=tk.X)
        self._reg(uva_grid, "card_frame")

        self.us_streaming_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(uva_grid, text="Acoustic Streaming (k boost in melt pool)", variable=self.us_streaming_var).grid(row=0, column=0, sticky=tk.W, padx=6, pady=3)

        self.us_heatgen_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(uva_grid, text="Acoustic Heat Generation (dissipation)", variable=self.us_heatgen_var).grid(row=0, column=1, sticky=tk.W, padx=6, pady=3)

        self.us_cavitation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(uva_grid, text="Acoustic Cavitation (void fraction)", variable=self.us_cavitation_var).grid(row=1, column=0, sticky=tk.W, padx=6, pady=3)

        self.us_momentum_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(uva_grid, text="Acoustic Momentum Transfer (radiation pressure)", variable=self.us_momentum_var).grid(row=1, column=1, sticky=tk.W, padx=6, pady=3)

        # --- Multi-Material Zones ---
        zone_lf = tk.LabelFrame(content, text="Multi-Material Zones (Layer-by-Layer Architecture)", font=FONT_SECTION, padx=10, pady=8)
        zone_lf.pack(fill=tk.X, pady=(0, 10))
        self._reg(zone_lf, "card_labelframe")
        self._add_info_btn_to_labelframe(zone_lf, self._show_ultrasonic_info)

        top_bar = tk.Frame(zone_lf)
        top_bar.pack(fill=tk.X, pady=(0, 6))
        self._reg(top_bar, "card_frame")

        tk.Label(
            top_bar, text="Define discrete material zones, layer allocations, and build sequence:",
            font=FONT_SMALL, fg="#4b5563"
        ).pack(side=tk.LEFT)

        btn_box = tk.Frame(top_bar)
        btn_box.pack(side=tk.RIGHT)
        self._reg(btn_box, "card_frame")

        add_btn = tk.Button(
            btn_box, text="+ Add Zone", font=FONT_SMALL_BOLD,
            bg=self.colors.get("button_primary", "#2563eb"),
            fg=self.colors.get("button_primary_fg", "#ffffff"),
            activebackground=self.colors.get("button_primary_active", "#1d4ed8"),
            relief=tk.FLAT, padx=10, pady=3, cursor="hand2",
            command=self._add_us_zone
        )
        add_btn.pack(side=tk.RIGHT, padx=(4, 0))
        self._reg(add_btn, "button")

        sync_btn = tk.Button(
            btn_box, text="Auto-Balance", font=FONT_SMALL,
            relief=tk.FLAT, padx=8, pady=3, cursor="hand2",
            command=self._auto_balance_us_zones
        )
        sync_btn.pack(side=tk.RIGHT)
        self._reg(sync_btn, "button")

        # Container frame for zone rows
        self.us_zones_container = tk.Frame(zone_lf)
        self.us_zones_container.pack(fill=tk.X, pady=2)
        self._reg(self.us_zones_container, "card_frame")

        self.us_zone_entries = []

        # Summary label
        self.us_zones_summary_lbl = tk.Label(zone_lf, text="", font=FONT_SMALL_BOLD, fg="#1d4ed8")
        self.us_zones_summary_lbl.pack(anchor=tk.W, pady=(6, 0))
        self._reg(self.us_zones_summary_lbl, "label")

        # Initialize default 2 zones (Zone 1: 5 layers SS308L, Zone 2: 5 layers SS316L)
        self._add_us_zone(layers=5, material="SS308L")
        self._add_us_zone(layers=5, material="SS316L")

        # Hook trace on total layers, bead width, bead height, and remelt ratio to sync geometry and refresh zone ranges
        try:
            self.us_layers_var.trace_add("write", lambda *args: (self._sync_ultrasonic_to_geometry(), self._refresh_us_zones()))
            self.us_width_var.trace_add("write", lambda *args: self._sync_ultrasonic_to_geometry())
            self.us_height_var.trace_add("write", lambda *args: (self._sync_ultrasonic_to_geometry(), self._refresh_us_zones()))
            self.us_remelt_var.trace_add("write", lambda *args: (self._sync_ultrasonic_to_geometry(), self._refresh_us_zones()))
        except Exception:
            pass

        # Run Button
        action_box = tk.LabelFrame(content, text="Simulation Action", font=FONT_SECTION, padx=10, pady=8)
        action_box.pack(fill=tk.X, pady=(0, 10))
        self._reg(action_box, "card_labelframe")
        self._add_info_btn_to_labelframe(action_box, self._show_ultrasonic_info)

        b1 = tk.Button(
            action_box, text="Run Ultrasonic Multi-Material Simulation",
            font=FONT_STATUS,
            bg=self.colors["button_primary"], fg=self.colors["button_primary_fg"],
            activebackground=self.colors["button_primary_active"],
            command=self._run_simulation, padx=10, pady=6
        )
        b1.pack(fill=tk.X, pady=3)
        self.claude_b1 = b1

        # Post-Processing & Output Links
        post_box = tk.LabelFrame(content, text="Results & Post-Processing", font=FONT_SECTION, padx=10, pady=8)
        post_box.pack(fill=tk.X, pady=(0, 10))
        self._reg(post_box, "card_labelframe")
        self._add_info_btn_to_labelframe(post_box, self._show_ultrasonic_info)

        prow = tk.Frame(post_box)
        prow.pack(fill=tk.X)
        self._reg(prow, "card_frame")

        pv_btn = tk.Button(
            prow, text="Open in ParaView (waam.pvd)", font=FONT_BTN,
            command=self._open_paraview, padx=10, pady=4
        )
        pv_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.claude_pv_btn2 = pv_btn
        self._reg(pv_btn, "button")

        out_btn = tk.Button(
            prow, text="Open Output Directory", font=FONT_BTN,
            command=self._open_output, padx=10, pady=4
        )
        out_btn.pack(side=tk.LEFT)
        self.claude_out_btn2 = out_btn
        self._reg(out_btn, "button")

        self._update_ultrasonic_status_badge()

    def _on_ultrasonic_physics_toggle(self):
        """Update ultrasonic status badge when physics checkbox is toggled."""
        self._update_ultrasonic_status_badge()

    def _toggle_ultrasonic_from_tab(self):
        """Toggle ultrasonic physics checkbox directly from Ultrasonic tab."""
        if hasattr(self, "enable_ultrasonic_var"):
            new_val = not self.enable_ultrasonic_var.get()
            self.enable_ultrasonic_var.set(new_val)
            self._update_ultrasonic_status_badge()

    def _update_ultrasonic_status_badge(self):
        """Refresh badge in Ultrasonic tab reflecting whether physics is enabled."""
        if hasattr(self, "us_physics_status_lbl") and hasattr(self, "enable_ultrasonic_var"):
            is_on = bool(self.enable_ultrasonic_var.get())
            if is_on:
                self.us_physics_status_lbl.config(
                    text="✓ Ultrasonic Physics Module: ACTIVE (Coupled 20 kHz Acoustic Softening ON)",
                    fg="#16a34a"
                )
                if hasattr(self, "us_toggle_btn"):
                    self.us_toggle_btn.config(text="Disable in Physics Tab", fg="#dc2626")
            else:
                self.us_physics_status_lbl.config(
                    text="⚠ Ultrasonic Physics Module: DISABLED (Physics Tab OFF — Zero Acoustic Softening Applied)",
                    fg="#ea580c"
                )
                if hasattr(self, "us_toggle_btn"):
                    self.us_toggle_btn.config(text="Enable in Physics Tab", fg="#2563eb")

    def _add_us_zone(self, layers: int = None, material: str = None):
        """Add a new configurable material zone row in the Ultrasonic tab."""
        mat_options = ["SS308L", "SS316L", "Ti-6Al-4V", "Pure Al (Al1xxx)"]
        
        # Calculate sensible default layers if not specified
        if layers is None:
            try:
                tot_req = int(self.us_layers_var.get())
            except Exception:
                tot_req = 10
            cur_sum = 0
            for e in getattr(self, "us_zone_entries", []):
                val = e["layers_var"].get().strip()
                if val.isdigit():
                    cur_sum += int(val)
            rem = tot_req - cur_sum
            layers = rem if rem > 0 else 5

        # Sensible default material
        if material is None:
            idx = len(getattr(self, "us_zone_entries", []))
            material = mat_options[idx % len(mat_options)]

        row_frame = tk.Frame(self.us_zones_container, pady=3)
        row_frame.pack(fill=tk.X)
        self._reg(row_frame, "card_frame")

        lbl_zone = tk.Label(row_frame, text=f"Zone {len(self.us_zone_entries) + 1}:", font=FONT_SMALL_BOLD, width=9, anchor=tk.W)
        lbl_zone.pack(side=tk.LEFT, padx=(2, 6))
        self._reg(lbl_zone, "label")

        tk.Label(row_frame, text="Layers:", font=FONT_SMALL).pack(side=tk.LEFT, padx=(2, 2))
        layers_var = tk.StringVar(value=str(layers))
        entry_layers = ttk.Entry(row_frame, textvariable=layers_var, width=6)
        entry_layers.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(row_frame, text="Material:", font=FONT_SMALL).pack(side=tk.LEFT, padx=(4, 2))
        mat_var = tk.StringVar(value=material)
        combo_mat = ttk.Combobox(row_frame, textvariable=mat_var, values=mat_options, state="readonly", width=12)
        combo_mat.pack(side=tk.LEFT, padx=(0, 8))

        lbl_range = tk.Label(row_frame, text="", font=FONT_MONO_SM, fg="#4b5563")
        lbl_range.pack(side=tk.LEFT, padx=(4, 8))
        self._reg(lbl_range, "label")

        del_btn = tk.Button(
            row_frame, text="✕", font=("TkDefaultFont", 9, "bold"),
            fg="#dc2626", relief=tk.FLAT, padx=4, pady=0, cursor="hand2",
            command=lambda rf=row_frame: self._remove_us_zone(rf)
        )
        del_btn.pack(side=tk.LEFT)
        self._reg(del_btn, "button")

        entry_dict = {
            "frame": row_frame,
            "lbl_zone": lbl_zone,
            "layers_var": layers_var,
            "mat_var": mat_var,
            "mat_combo": combo_mat,
            "lbl_range": lbl_range,
            "del_btn": del_btn
        }
        self.us_zone_entries.append(entry_dict)

        try:
            layers_var.trace_add("write", lambda *args: self._refresh_us_zones())
            mat_var.trace_add("write", lambda *args: self._refresh_us_zones())
        except Exception:
            pass
        self._refresh_us_zones()

    def _remove_us_zone(self, row_frame: tk.Frame):
        """Remove a zone row if more than 1 zone remains."""
        if len(self.us_zone_entries) <= 1:
            return
        idx_to_remove = None
        for i, entry in enumerate(self.us_zone_entries):
            if entry["frame"] == row_frame:
                idx_to_remove = i
                break
        if idx_to_remove is not None:
            entry = self.us_zone_entries.pop(idx_to_remove)
            entry["frame"].destroy()
            self._refresh_us_zones()

    def _auto_balance_us_zones(self):
        """Evenly distribute total layers across all defined zones."""
        if not getattr(self, "us_zone_entries", []):
            return
        try:
            total_req = int(self.us_layers_var.get())
        except Exception:
            total_req = 10
        n_zones = len(self.us_zone_entries)
        if n_zones <= 0:
            return
        base_per_zone = max(1, total_req // n_zones)
        rem = total_req - base_per_zone * n_zones
        for i, entry in enumerate(self.us_zone_entries):
            extra = 1 if i < rem else 0
            entry["layers_var"].set(str(base_per_zone + extra))
        self._refresh_us_zones()

    def _refresh_us_zones(self):
        """Update zone labels, Z-height ranges, delete button states, and summary bar."""
        if not hasattr(self, "us_zone_entries"):
            return
        try:
            tot_req = int(self.us_layers_var.get())
        except Exception:
            tot_req = 10
        try:
            h_bead_first = float(self.us_height_var.get())
        except Exception:
            h_bead_first = 4.0
        try:
            remelt_pct = float(self.us_remelt_var.get())
            remelt_ratio = (remelt_pct / 100.0) if remelt_pct > 1.0 else remelt_pct
        except Exception:
            remelt_ratio = 0.80

        h_bead_subseq = h_bead_first * remelt_ratio
        if hasattr(self, "us_fusion_info_lbl"):
            self.us_fusion_info_lbl.config(
                text=f"Subsequent: {h_bead_subseq:.2f} mm ({remelt_ratio*100.0:.0f}% fusion)"
            )

        def calc_z(n):
            if n <= 0:
                return 0.0
            elif n == 1:
                return h_bead_first
            else:
                return h_bead_first + (n - 1) * h_bead_subseq

        cum_l = 0
        can_delete = len(self.us_zone_entries) > 1

        for i, entry in enumerate(self.us_zone_entries):
            entry["lbl_zone"].config(text=f"Zone {i + 1}:")
            val_str = entry["layers_var"].get().strip()
            l_count = int(val_str) if val_str.isdigit() else 0
            
            start_l = cum_l + 1 if l_count > 0 else cum_l
            end_l = cum_l + l_count
            start_z = calc_z(cum_l)
            end_z = calc_z(cum_l + l_count)
            cum_l += l_count

            if l_count > 0:
                entry["lbl_range"].config(text=f"Layers {start_l}–{end_l} (Z: {start_z:.1f}–{end_z:.1f} mm)")
            else:
                entry["lbl_range"].config(text="0 layers")

            if can_delete:
                entry["del_btn"].config(state=tk.NORMAL)
            else:
                entry["del_btn"].config(state=tk.DISABLED)

        # Update summary label
        total_alloc_z = calc_z(cum_l)
        if hasattr(self, "us_zones_summary_lbl"):
            if cum_l == tot_req:
                self.us_zones_summary_lbl.config(
                    text=f"✓ All Layers Allocated: {cum_l} / {tot_req} layers  |  Total Build Height: {total_alloc_z:.1f} mm",
                    fg="#16a34a"
                )
            elif cum_l < tot_req:
                self.us_zones_summary_lbl.config(
                    text=f"⚠ Under-allocated: {cum_l} / {tot_req} layers ({tot_req - cum_l} layers remaining)  |  Height: {total_alloc_z:.1f} mm",
                    fg="#d97706"
                )
            else:
                self.us_zones_summary_lbl.config(
                    text=f"⚠ Over-allocated: {cum_l} / {tot_req} layers ({cum_l - tot_req} extra layers)  |  Height: {total_alloc_z:.1f} mm",
                    fg="#dc2626"
                )

    def _build_sim_log_tab(self):
        """Dedicated Simulation Log Viewer Tab with search, color tags, and export."""
        self.sim_log_tab = tk.Frame(self.notebook)
        self.notebook.add(self.sim_log_tab, text="Simulation Log")
        self._reg(self.sim_log_tab, "card_frame")

        # Toolbar
        toolbar = tk.Frame(self.sim_log_tab, pady=4)
        toolbar.pack(fill=tk.X, padx=6)
        self._reg(toolbar, "card_frame")

        tk.Label(toolbar, text="Search:", font=FONT_LABEL).pack(side=tk.LEFT, padx=(0, 4))
        self._reg(toolbar.winfo_children()[-1], "label")
        self.log_search_var = tk.StringVar()
        search_entry = tk.Entry(toolbar, textvariable=self.log_search_var, width=24, font=FONT_ENTRY, relief=tk.SOLID, bd=1)
        search_entry.pack(side=tk.LEFT, padx=2)
        self._reg(search_entry, "entry")

        find_btn = tk.Button(toolbar, text="Find", font=FONT_BTN, command=self._log_search, padx=6, pady=2)
        find_btn.pack(side=tk.LEFT, padx=2)
        self._reg(find_btn, "button")

        reload_btn = tk.Button(toolbar, text="Reload Log", font=FONT_BTN, command=self._log_reload, padx=6, pady=2)
        reload_btn.pack(side=tk.LEFT, padx=2)
        self._reg(reload_btn, "button")

        export_btn = tk.Button(toolbar, text="Export Log", font=FONT_BTN, command=self._log_export, padx=6, pady=2)
        export_btn.pack(side=tk.LEFT, padx=2)
        self._reg(export_btn, "button")

        clear_btn = tk.Button(toolbar, text="Clear Display", font=FONT_BTN, command=self._log_clear, padx=6, pady=2)
        clear_btn.pack(side=tk.LEFT, padx=2)
        self._reg(clear_btn, "button")

        self.log_count_var = tk.StringVar(value="")
        log_count_lbl = tk.Label(toolbar, textvariable=self.log_count_var, font=FONT_SMALL)
        log_count_lbl.pack(side=tk.LEFT, padx=8)
        self._reg(log_count_lbl, "label_dim")

        # Log viewer text widget (CHAR wrap preserves long solver paths)
        self.log_viewer = scrolledtext.ScrolledText(
            self.sim_log_tab, font=FONT_CONSOLE, wrap=tk.CHAR, state=tk.DISABLED
        )
        self.log_viewer.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))

        # Color tags for log levels (re-themed in _apply_theme for contrast).
        # "warning" is an alias so both [WARN] and [WARNING] highlight.
        self.log_viewer.tag_configure("error",   foreground="#ef4444", font=(FONT_CONSOLE[0], FONT_CONSOLE[1], "bold"))
        self.log_viewer.tag_configure("warn",    foreground="#f59e0b", font=(FONT_CONSOLE[0], FONT_CONSOLE[1], "bold"))
        self.log_viewer.tag_configure("warning", foreground="#f59e0b", font=(FONT_CONSOLE[0], FONT_CONSOLE[1], "bold"))
        self.log_viewer.tag_configure("ok",      foreground="#22c55e")
        self.log_viewer.tag_configure("info",    foreground="#60a5fa")
        self.log_viewer.tag_configure("search",  background="#facc15", foreground="#000000")

    def _log_reload(self):
        """Reload sim.log from the FreeFEM side (legacy flat layout as fallback)."""
        run_dir = self._get_active_or_latest_run_dir()
        candidates = [
            os.path.join(run_dir, "FreeFEM", "sim.log"),
            os.path.join(run_dir, "FreeFEM", "simulation.log"),
            os.path.join(run_dir, "sim.log"),
            os.path.join(run_dir, "simulation.log"),
        ]
        log_path = next((c for c in candidates if os.path.exists(c)), candidates[0])
        if not os.path.exists(log_path):
            base = self._get_base_output_dir()
            for cand in (os.path.join(base, "sim.log"), os.path.join(base, "simulation.log")):
                if os.path.exists(cand):
                    log_path = cand
                    break

        self.log_viewer.config(state=tk.NORMAL)
        self.log_viewer.delete("1.0", tk.END)
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    tag = None
                    if "[ERROR]" in line or "[Error]" in line:
                        tag = "error"
                    elif "[WARN]" in line or "[WARNING]" in line or "[Warning]" in line:
                        tag = "warn"
                    elif "[OK]" in line or "Normal End" in line:
                        tag = "ok"
                    elif "[INFO]" in line:
                        tag = "info"
                    self.log_viewer.insert(tk.END, line, tag)
            self.log_viewer.see(tk.END)
        except FileNotFoundError:
            self.log_viewer.insert(tk.END, f"[LOG] No simulation log found at: {os.path.abspath(log_path)}\n"
                                           f"Run a simulation first to generate output.\n")
        self.log_viewer.config(state=tk.DISABLED)

    def _log_search(self):
        """Highlight all occurrences of the search term in the log viewer."""
        self.log_viewer.tag_remove("search", "1.0", tk.END)
        query = self.log_search_var.get().strip()
        if not query:
            if hasattr(self, "log_count_var"):
                self.log_count_var.set("")
            return
        count = 0
        start = "1.0"
        first = None
        while True:
            pos = self.log_viewer.search(query, start, stopindex=tk.END, nocase=True)
            if not pos:
                break
            if first is None:
                first = pos
            end = f"{pos}+{len(query)}c"
            self.log_viewer.tag_add("search", pos, end)
            count += 1
            start = end
        if hasattr(self, "log_count_var"):
            self.log_count_var.set(f"{count} match{'es' if count != 1 else ''}" if count else "No matches")
        if first:
            self.log_viewer.see(first)

    def _log_export(self):
        """Export the current log content to a user-chosen file."""
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("Log Files", "*.log"), ("All Files", "*.*")],
            title="Export Simulation Log"
        )
        if path:
            content = self.log_viewer.get("1.0", tk.END)
            with open(path, "w") as f:
                f.write(content)
            self._log(f"[LOG] Log exported to: {path}\n")

    def _log_clear(self):
        """Clear the log viewer content (with confirm)."""
        if not messagebox.askyesno("Clear Log", "Clear the Simulation Log view?\nLog files on disk are kept."):
            return
        self.log_viewer.config(state=tk.NORMAL)
        self.log_viewer.delete("1.0", tk.END)
        self.log_viewer.config(state=tk.DISABLED)

    def _console_clear(self):
        """Clear the active solver output stream console."""
        self.console.config(state=tk.NORMAL)
        self.console.delete("1.0", tk.END)
        self.console.config(state=tk.DISABLED)

    def _console_copy(self):
        """Copy active solver output stream contents to system clipboard."""
        try:
            content = self.console.get("1.0", tk.END).strip()
            if content:
                self.root.clipboard_clear()
                self.root.clipboard_append(content)
                if hasattr(self, "console_copy_btn"):
                    old_text = self.console_copy_btn.cget("text")
                    self.console_copy_btn.config(text="✓ Copied")
                    self.root.after(1500, lambda: self.console_copy_btn.config(text=old_text))
        except Exception:
            pass

    def _init_console_tags(self):
        """Configure syntax tags for solver output stream highlighting and typography."""
        if not hasattr(self, "console"):
            return
        is_dark = (getattr(self, "current_theme", "light") == "dark")
        tag_colors = {
            "banner": {
                "foreground": "#38bdf8" if is_dark else "#0284c7",
                "font": (FONT_CONSOLE[0], FONT_CONSOLE[1], "bold")
            },
            "layer": {
                "foreground": "#67e8f9" if is_dark else "#0369a1",
                "font": (FONT_CONSOLE[0], FONT_CONSOLE[1], "bold"),
                "spacing1": 5
            },
            "phase": {
                "foreground": "#c084fc" if is_dark else "#7c3aed",
                "font": (FONT_CONSOLE[0], FONT_CONSOLE[1], "bold"),
                "spacing1": 4
            },
            "step": {
                "foreground": "#cbd5e1" if is_dark else "#334155"
            },
            "ok": {
                "foreground": "#4ade80" if is_dark else "#16a34a",
                "font": (FONT_CONSOLE[0], FONT_CONSOLE[1], "bold")
            },
            "warn": {
                "foreground": "#fbbf24" if is_dark else "#d97706",
                "font": (FONT_CONSOLE[0], FONT_CONSOLE[1], "bold")
            },
            "error": {
                "foreground": "#f87171" if is_dark else "#dc2626",
                "font": (FONT_CONSOLE[0], FONT_CONSOLE[1], "bold")
            },
            "info": {
                "foreground": "#38bdf8" if is_dark else "#0284c7"
            },
            "bullet": {
                "foreground": "#a78bfa" if is_dark else "#6366f1",
                "font": (FONT_CONSOLE[0], FONT_CONSOLE[1], "bold")
            },
            "dim": {
                "foreground": "#64748b"
            },
        }
        for tag_name, opts in tag_colors.items():
            try:
                self.console.tag_configure(tag_name, **opts)
            except Exception:
                pass


    def _refresh_embedded_viewport(self):
        """Update the embedded 3D mesh viewport with current parameters."""
        try:
            params = self._collect_params()
            if hasattr(self, "loaded_stl_path") and self.loaded_stl_path and os.path.exists(self.loaded_stl_path):
                params["stl_file"] = self.loaded_stl_path
            if hasattr(self, "mesh_file_var") and self.mesh_file_var.get():
                mpath = self.mesh_file_var.get().strip()
                if os.path.exists(mpath):
                    if mpath.lower().endswith(('.gcode', '.nc')):
                        params["gcode_file"] = mpath
                    else:
                        params["mesh_file"] = mpath
            if hasattr(self, "viewport"):
                self.viewport.update_mesh(params)
            elif hasattr(self, "viewport_3d"):
                self.viewport_3d.update_mesh(params)
        except Exception:
            pass

    def _mark_custom_preset(self):
        """Auto-switch the preset selector to Custom on any manual edit.

        Programmatic .set() on preset_var does NOT fire the combobox
        <<ComboboxSelected>> event, so this never re-triggers _apply_preset.
        Suppressed while _apply_preset itself is writing values.
        """
        try:
            if getattr(self, "_applying_preset", False):
                return
            if not hasattr(self, "preset_var"):
                return
            if "Custom" not in PARAMETER_PRESETS:
                return
            if self.preset_var.get() != "Custom":
                self.preset_var.set("Custom")
        except Exception:
            pass

    def _apply_preset(self, event=None):
        """Apply a named parameter preset to all relevant GUI fields."""
        name = self.preset_var.get()
        # "Custom" bypasses every hardcoded preset: keep UI values verbatim.
        if name == "Custom":
            try:
                self._log("[PRESET] Custom configuration: using current UI values verbatim (no preset override).\n")
            except Exception:
                pass
            return
        preset = PARAMETER_PRESETS.get(name)
        if not preset:
            return

        self._applying_preset = True
        try:
            # 1. Heat source first so its dimensions widgets exist
            if "heat_source" in preset and hasattr(self, "source_var"):
                self.source_var.set(str(preset["heat_source"]))
                self._update_source_params()

            # 2. Materials
            if "material_base" in preset and hasattr(self, "material_base_var"):
                self.material_base_var.set(str(preset["material_base"]))
            if "material_dep" in preset and hasattr(self, "material_dep_var"):
                self.material_dep_var.set(str(preset["material_dep"]))

            # 3. Geometry & Mesh source mode
            if "mesh_source" in preset and hasattr(self, "mesh_source_var"):
                self.mesh_source_var.set(str(preset["mesh_source"]))
                if hasattr(self, "_update_mesh_ui"):
                    self._update_mesh_ui()

            # 4. Mechanical BC option
            if "mechanical_bc" in preset and hasattr(self, "mech_bc_var"):
                bc_val = str(preset["mechanical_bc"]).lower()
                if "cantilever" in bc_val:
                    self.mech_bc_var.set("Cantilever (x=0 Face Fixed)")
                elif "clamped" in bc_val or "fixture" in bc_val:
                    self.mech_bc_var.set("Clamped Fixture (Full Baseplate Fixed)")
                else:
                    self.mech_bc_var.set("Isostatic 3-2-1 / Elastic Support (Recommended)")

            # 5. Physics checkboxes
            if "enable_radiation" in preset and hasattr(self, "rad_var"):
                self.rad_var.set(bool(preset["enable_radiation"]))
            if "enable_latent" in preset and hasattr(self, "latent_var"):
                self.latent_var.set(bool(preset["enable_latent"]))
            if "enable_mechanical" in preset and hasattr(self, "mech_var"):
                self.mech_var.set(bool(preset["enable_mechanical"]))
            if "enable_microstructure" in preset and hasattr(self, "micro_var"):
                self.micro_var.set(bool(preset["enable_microstructure"]))

            if hasattr(self, "_on_multi_material_change"):
                self._on_multi_material_change()

            # 6. Parameters in _param_vars
            for key, val in preset.items():
                if key in ("material_base", "material_dep", "heat_source", "mesh_source", "mechanical_bc",
                           "enable_radiation", "enable_latent", "enable_mechanical", "enable_microstructure"):
                    continue
                if key in self._param_vars:
                    self._param_vars[key][0].set("" if val is None else str(val))
                self.params[key] = val

        finally:
            self._applying_preset = False

        # A preset rewrites N_layer / h_bead / remelt: re-apply the active
        # dimensioning driver so H_build and N_layer can never go stale.
        try:
            self._sync_layer_mode()
        except Exception:
            pass
        self._update_mesh_calc()
        self._update_heat_calc()
        try:
            self._update_geom_fusion_label()
        except Exception:
            pass
        try:
            self._auto_sync_probe_points()
        except Exception:
            pass
        try:
            self._auto_update_wall_gcode()
        except Exception:
            pass
        self._log(f"[PRESET] Applied preset: {name}\n")

    def _toggle_theme(self):
        """Toggle between light and dark themes."""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.colors = THEMES[self.current_theme]
        self._apply_theme()

    def _apply_theme(self):
        """Apply colors of the active theme to all UI elements."""
        if not hasattr(self, "btn_bar"):
            return

        c = self.colors
        bg = c["bg"]
        card = c["bg_card"]
        fg = c["fg"]
        accent = c["accent"]
        border = c["border"]

        self.root.configure(bg=bg)
        self.header.configure(bg=bg)
        self.title_frame.configure(bg=bg)
        self.header_right.configure(bg=bg)
        self.title_label.configure(bg=bg, fg=accent)
        if hasattr(self, "subtitle_label") and self.subtitle_label:
            self.subtitle_label.configure(bg=bg, fg=c["fg_dim"])
        self.status_label.configure(bg=bg, fg=c["accent2"])
        self.theme_btn.configure(
            bg=c["button"], fg=c["button_fg"],
            text="[ Dark Theme ]" if self.current_theme == "light" else "[ Light Theme ]"
        )
        self.right_panel.configure(bg=card)
        self.btn_bar.configure(bg=card)
        self.main_pane.configure(bg=bg)
        self.console_frame.configure(bg=card, fg=accent)
        if hasattr(self, "console_toolbar"):
            self.console_toolbar.configure(bg=card)
        if hasattr(self, "console_status_dot"):
            self.console_status_dot.configure(bg=card)
        if hasattr(self, "console_clear_btn"):
            self.console_clear_btn.configure(bg=c["button"], fg=c["button_fg"])
        if hasattr(self, "console_copy_btn"):
            self.console_copy_btn.configure(bg=c["button"], fg=c["button_fg"])
        self.console.configure(
            bg=c["console_bg"], fg=c["console_fg"],
            insertbackground=c["console_fg"]
        )
        self._init_console_tags()

        # Style standard ttk elements
        self.style.configure(".", background=bg, foreground=fg)
        self.style.configure("TNotebook", background=bg)
        self.style.configure(
            "TNotebook.Tab",
            background=c["tab_unselected_bg"],
            foreground=c["tab_unselected_fg"],
            padding=[10, 5],
            font=FONT_TAB
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", c["tab_selected_bg"])],
            foreground=[("selected", c["tab_selected_fg"])]
        )

        # Action buttons - all theme-aware (no hardcoded hex).
        # Standard grey buttons follow the theme button palette.
        _std_btns = [self.gen_btn, self.out_btn, self.pv_btn, self.foam_btn, self.pv_gpu_btn]
        for _extra in ("claude_100mm_btn", "claude_pv_btn"):
            if hasattr(self, _extra):
                _std_btns.append(getattr(self, _extra))
        for btn in _std_btns:
            try:
                btn.configure(
                    bg=c["button"], fg=c["button_fg"],
                    activebackground=border, activeforeground=c["button_fg"],
                    highlightbackground=border, relief=tk.FLAT, bd=0, highlightthickness=0
                )
            except Exception:
                pass

        # Accent buttons keep their hue per theme (light saturated / dark pastel)
        try:
            self.view_results_btn.configure(
                bg=c["button_primary"], fg=c["button_primary_fg"],
                activebackground=c["button_primary_active"],
                activeforeground=c["button_primary_fg"],
                relief=tk.FLAT, bd=0, highlightthickness=0
            )
        except Exception:
            pass
        try:
            self.mesh3d_btn.configure(
                bg=c["button_info"], fg=c["button_info_fg"],
                activebackground=c["accent"],
                activeforeground=c["button_info_fg"],
                relief=tk.FLAT, bd=0, highlightthickness=0
            )
        except Exception:
            pass
        if hasattr(self, "claude_btn"):
            try:
                self.claude_btn.configure(
                    bg=c["button_violet"], fg=c["button_violet_fg"],
                    activebackground=c["button_violet_active"],
                    activeforeground=c["button_violet_fg"],
                    relief=tk.FLAT, bd=0, highlightthickness=0
                )
            except Exception:
                pass
        for _b in ("claude_b1", "claude_b2"):
            if hasattr(self, _b):
                try:
                    _w = getattr(self, _b)
                    _is_primary = (_b == "claude_b1")
                    _w.configure(
                        bg=c["button_primary"] if _is_primary else c["button_violet"],
                        fg=c["button_primary_fg"] if _is_primary else c["button_violet_fg"],
                        activebackground=c["button_primary_active"] if _is_primary else c["button_violet_active"],
                        relief=tk.FLAT, bd=0, highlightthickness=0
                    )
                except Exception:
                    pass
        # CFD + Claude tab action buttons stored as refs where available.
        for _b in ("cfd_gen_btn", "cfd_run_btn", "claude_b3", "claude_pv_btn2", "claude_out_btn2"):
            if hasattr(self, _b):
                try:
                    getattr(self, _b).configure(
                        bg=c["button"], fg=c["button_fg"],
                        activebackground=border, activeforeground=c["button_fg"],
                        relief=tk.FLAT, bd=0, highlightthickness=0
                    )
                except Exception:
                    pass

        self.run_btn.configure(
            bg=c["button_run"], fg=c["button_run_fg"],
            activebackground=c["accent2"], activeforeground=c["button_run_fg"],
            relief=tk.FLAT, bd=0, highlightthickness=0
        )
        self.stop_btn.configure(
            bg=c["error"], fg=c.get("button_run_fg", "#000000"),
            activebackground=border,
            relief=tk.FLAT, bd=0, highlightthickness=0
        )

        # Recursively apply to registered frames & labels.
        # NOTE: several Buttons were historically registered as "card_frame".
        # Treat those as theme buttons (bg + fg), otherwise text stays black
        # on a dark card and the theme looks "not changing".
        for widget, role in self.registered_widgets:
            try:
                if role == "card_frame":
                    try:
                        if widget.winfo_class() == "Button":
                            widget.configure(
                                bg=c["button"], fg=c["button_fg"],
                                activebackground=border, activeforeground=c["button_fg"],
                                highlightbackground=border,
                            )
                        else:
                            widget.configure(bg=card)
                    except Exception:
                        widget.configure(bg=card)
                elif role == "card_labelframe":
                    widget.configure(bg=card, fg=accent)
                elif role == "label":
                    widget.configure(bg=card, fg=fg)
                elif role == "label_dim":
                    widget.configure(bg=card, fg=c["fg_dim"])
                elif role == "entry":
                    widget.configure(
                        bg=c["bg_input"], fg=fg, insertbackground=fg,
                        readonlybackground=c["bg_input"], disabledbackground=c["bg_input"],
                        disabledforeground=c["fg_dim"], highlightbackground=border
                    )
                elif role == "button":
                    widget.configure(
                        bg=c["button"], fg=c["button_fg"],
                        activebackground=border, activeforeground=c["button_fg"],
                        highlightbackground=border,
                    )
                elif role == "radio":
                    widget.configure(bg=card, fg=fg, selectcolor=c["bg_input"], activebackground=card, activeforeground=accent)
                elif role == "check":
                    widget.configure(bg=card, fg=fg, selectcolor=c["bg_input"], activebackground=card)
            except Exception:
                pass

        # Text / Canvas / extra views that are otherwise stuck light.
        for _t in ("dissimilar_info", "default_case_text", "log_viewer"):
            if hasattr(self, _t):
                try:
                    getattr(self, _t).configure(
                        bg=c["bg_input"], fg=fg, insertbackground=fg,
                        highlightbackground=border, selectbackground=accent,
                    )
                except Exception:
                    pass
        # Scrollable tab canvases (Mesh / OpenFOAM) + viewport canvas frames.
        for _cv in ("mesh_canvas", "openfoam_canvas"):
            if hasattr(self, _cv):
                try:
                    getattr(self, _cv).configure(bg=card, highlightbackground=border)
                except Exception:
                    pass
        # Embedded 3D viewport follows the app theme (light/dark chrome).
        for _vp in ("viewport", "viewport_3d"):
            if hasattr(self, _vp):
                try:
                    _obj = getattr(self, _vp)
                    if hasattr(_obj, "apply_theme"):
                        _obj.apply_theme(self.current_theme)
                except Exception:
                    pass
        # Log level colors per theme (amber/green need dark variants for contrast).
        if hasattr(self, "log_viewer"):
            try:
                if self.current_theme == "dark":
                    self.log_viewer.tag_configure("error", foreground="#f38ba8")
                    self.log_viewer.tag_configure("warn", foreground="#f9e2af")
                    self.log_viewer.tag_configure("warning", foreground="#f9e2af")
                    self.log_viewer.tag_configure("ok", foreground="#a6e3a1")
                    self.log_viewer.tag_configure("info", foreground="#89b4fa")
                else:
                    self.log_viewer.tag_configure("error", foreground="#dc3545")
                    self.log_viewer.tag_configure("warn", foreground="#b45309")
                    self.log_viewer.tag_configure("warning", foreground="#b45309")
                    self.log_viewer.tag_configure("ok", foreground="#198754")
                    self.log_viewer.tag_configure("info", foreground="#0d6efd")
                self.log_viewer.tag_configure("search", background="#facc15", foreground="#1e1e2e")
            except Exception:
                pass

    def _reg(self, widget, role):
        self.registered_widgets.append((widget, role))
        return widget

    def _make_param_row(self, parent, label, key, row, unit="", type_="float", width=10, on_change=None):
        lbl = tk.Label(parent, text=label, font=FONT_LABEL)
        lbl.grid(row=row, column=0, sticky=tk.W, padx=(8, 4), pady=2)
        self._reg(lbl, "label")

        raw_val = self.params.get(key, "")
        if key in MM_DIMENSION_KEYS and isinstance(raw_val, (int, float)):
            display_val = round(raw_val * 1000.0, 2)
            raw_val = int(display_val) if display_val == int(display_val) else display_val
        elif key in MM_SPEED_KEYS and isinstance(raw_val, (int, float)):
            display_val = round(raw_val * 1000.0, 2)
            raw_val = int(display_val) if display_val == int(display_val) else display_val

        var = tk.StringVar(value=str(raw_val))
        entry = tk.Entry(parent, textvariable=var, width=width, font=FONT_ENTRY, relief=tk.SOLID, bd=1)
        entry.grid(row=row, column=1, padx=4, pady=2, sticky=tk.W)
        self._reg(entry, "entry")

        if unit:
            u_lbl = tk.Label(parent, text=unit, font=FONT_SMALL)
            u_lbl.grid(row=row, column=2, sticky=tk.W, padx=4)
            self._reg(u_lbl, "label_dim")

        self._param_vars[key] = (var, type_)
        if not hasattr(self, "param_entries"):
            self.param_entries = {}
        self.param_entries[key] = entry

        if key == "n_layers":
            entry.bind("<FocusIn>", lambda e: self._on_n_layers_focus())
            entry.bind("<KeyRelease>", lambda e: self._on_n_layers_key())

        if on_change:
            var.trace_add("write", lambda *args: on_change())
        else:
            # Central fan-out: EVERY parameter change refreshes every dependent readout
            var.trace_add("write", lambda *args: self._on_any_param_change(key))

        return var

    def _on_any_param_change(self, key):
        """Single refresh path for any parameter edit (window A -> window B)."""
        try:
            self._mark_custom_preset()
            if key == "n_layers":
                if not getattr(self, "_syncing_layer_mode", False):
                    if hasattr(self, "_layer_mode") and self._layer_mode.get() != "n_layers":
                        self._layer_mode.set("n_layers")
                    self._sync_from_n_layers()
            elif key in ("h_bead", "remelt_ratio"):
                self._sync_layer_mode()
            if key in ("n_layers", "h_bead", "remelt_ratio", "ty_dep"):
                self._sync_geometry_to_ultrasonic()
            if key == "n_layers" and "nz_dep" in self._param_vars:
                nl_v = self._param_vars["n_layers"][0].get().strip()
                nz_v = self._param_vars["nz_dep"][0].get().strip()
                prev_nl = getattr(self, "_last_n_layers", None)
                # The mesh tab couples nz_dep to n_layers (1 cell/layer), but an
                # explicit user edit used to be silently overwritten on the next
                # keystroke. Only re-sync while the entry still tracks the
                # previous layer count; a diverged value is the user's mesh
                # resolution choice and is respected (the builder sizes the
                # source against the effective cells/layer either way).
                if nl_v and (nz_v == "" or prev_nl is None or nz_v == str(prev_nl)):
                    if nz_v != nl_v:
                        self._param_vars["nz_dep"][0].set(nl_v)
                try:
                    self._last_n_layers = int(nl_v)
                except (TypeError, ValueError):
                    pass
            self._update_mesh_calc()
            self._update_heat_calc()
            self._update_geom_fusion_label()
            self._update_material_wire_readout()
            self._auto_sync_probe_points()
            self._auto_update_wall_gcode()
        except Exception:
            pass

    def _sync_layer_mode(self):
        """Apply whichever of N_layer / target build height is currently the driver."""
        if not hasattr(self, "_layer_mode"):
            return
        if self._layer_mode.get() == "n_layers":
            self._sync_from_n_layers()
        else:
            self._sync_from_h_bead()

    def _sync_geometry_to_ultrasonic(self):
        """Keep Ultrasonic tab parameters in sync with Geometry tab."""
        if not hasattr(self, "_param_vars") or getattr(self, "_syncing_geom_us", False):
            return
        self._syncing_geom_us = True
        try:
            if hasattr(self, "us_layers_var") and "n_layers" in self._param_vars:
                nl_val = self._param_vars["n_layers"][0].get().strip()
                if nl_val and self.us_layers_var.get() != nl_val:
                    self.us_layers_var.set(nl_val)
                    if hasattr(self, "_auto_balance_us_zones"):
                        self._auto_balance_us_zones()
            if hasattr(self, "us_width_var") and "ty_dep" in self._param_vars:
                w_val = self._param_vars["ty_dep"][0].get().strip()
                if w_val and self.us_width_var.get() != w_val:
                    self.us_width_var.set(w_val)
            if hasattr(self, "us_height_var") and "h_bead" in self._param_vars:
                h_val = self._param_vars["h_bead"][0].get().strip()
                if h_val and self.us_height_var.get() != h_val:
                    self.us_height_var.set(h_val)
            if hasattr(self, "us_remelt_var") and "remelt_ratio" in self._param_vars:
                r_raw = self._param_vars["remelt_ratio"][0].get().strip()
                try:
                    r_flt = float(r_raw)
                    r_pct = str(round(r_flt * 100.0, 1)) if r_flt <= 1.0 else str(round(r_flt, 1))
                    if self.us_remelt_var.get() != r_pct:
                        self.us_remelt_var.set(r_pct)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._syncing_geom_us = False

    def _sync_ultrasonic_to_geometry(self):
        """Keep Geometry tab in sync if Ultrasonic tab entries are edited."""
        if not hasattr(self, "_param_vars") or getattr(self, "_syncing_geom_us", False):
            return
        # Only permit Ultrasonic tab to drive Geometry tab when Ultrasonic physics is actually enabled
        is_us = getattr(self, "enable_ultrasonic_var", None)
        if is_us is not None and not is_us.get():
            return
        self._syncing_geom_us = True
        try:
            if hasattr(self, "us_layers_var") and "n_layers" in self._param_vars:
                u_nl = self.us_layers_var.get().strip()
                if u_nl and self._param_vars["n_layers"][0].get() != u_nl:
                    self._param_vars["n_layers"][0].set(u_nl)
            if hasattr(self, "us_width_var") and "ty_dep" in self._param_vars:
                u_w = self.us_width_var.get().strip()
                if u_w and self._param_vars["ty_dep"][0].get() != u_w:
                    self._param_vars["ty_dep"][0].set(u_w)
            if hasattr(self, "us_height_var") and "h_bead" in self._param_vars:
                u_h = self.us_height_var.get().strip()
                if u_h and self._param_vars["h_bead"][0].get() != u_h:
                    self._param_vars["h_bead"][0].set(u_h)
            if hasattr(self, "us_remelt_var") and "remelt_ratio" in self._param_vars:
                u_r = self.us_remelt_var.get().strip()
                try:
                    r_flt = float(u_r)
                    ratio_str = str(round(r_flt / 100.0, 3)) if r_flt > 1.0 else str(round(r_flt, 3))
                    if self._param_vars["remelt_ratio"][0].get() != ratio_str:
                        self._param_vars["remelt_ratio"][0].set(ratio_str)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._syncing_geom_us = False

    # ============================================================
    # TABS IMPLEMENTATION
    # ============================================================

    def _build_geometry_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Geometry")
        self._reg(tab, "card_frame")

        lf1 = tk.LabelFrame(tab, text="Substrate (Base Plate) Dimensions", font=FONT_SECTION, padx=8, pady=4)
        lf1.pack(fill=tk.X, padx=8, pady=4)
        self._reg(lf1, "card_labelframe")
        self._add_info_btn_to_labelframe(lf1, self._show_substrate_info)
        self._make_param_row(lf1, "Substrate Length (Lₓ):", "lx", 0, "mm")
        self._make_param_row(lf1, "Substrate Width (W_sub):", "ly_base", 1, "mm")
        self._make_param_row(lf1, "Substrate Thickness (H_sub):", "hz_base", 2, "mm")

        lf2 = tk.LabelFrame(tab, text="WAAM Deposit Wall Geometry & Fusion Model", font=FONT_SECTION, padx=8, pady=4)
        lf2.pack(fill=tk.X, padx=8, pady=4)
        self._reg(lf2, "card_labelframe")
        self._add_info_btn_to_labelframe(lf2, self._show_fusion_info)
        self._make_param_row(lf2, "Single Bead Width (W_bead):", "ty_dep", 0, "mm")
        self._make_param_row(lf2, "First Layer Height (h₁):", "h_bead", 1, "mm")
        self._make_param_row(lf2, "Remelt / Fusion Ratio (η_remelt):", "remelt_ratio", 2, "[-]")
        # Dimensioning Mode Strategy (Explicit User Driver: Total Layers vs Target Wall Height)
        self._layer_mode = tk.StringVar(value="n_layers")
        mode_frame = tk.Frame(lf2)
        mode_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=8, pady=(6, 4))
        self._reg(mode_frame, "card_frame")

        lbl_mode = tk.Label(mode_frame, text="Dimensioning Driver:", font=FONT_SMALL_BOLD)
        lbl_mode.pack(side=tk.LEFT, padx=(0, 8))
        self._reg(lbl_mode, "label")

        rb_n = tk.Radiobutton(
            mode_frame, text="Define by Total Layers (N_layer)",
            variable=self._layer_mode, value="n_layers",
            font=FONT_SMALL, command=self._on_layer_mode_change
        )
        rb_n.pack(side=tk.LEFT, padx=6)
        self._reg(rb_n, "radio")

        rb_h = tk.Radiobutton(
            mode_frame, text="Define by Target Wall Height (H_build)",
            variable=self._layer_mode, value="height",
            font=FONT_SMALL, command=self._on_layer_mode_change
        )
        rb_h.pack(side=tk.LEFT, padx=6)
        self._reg(rb_h, "radio")

        # Row 4: Deposited Layers Count (N_layer)
        self._make_param_row(lf2, "Deposited Layers Count (N_layer):", "n_layers", 4, "layers", type_="int")

        # Row 5: Target Wall Build Height (H_build)
        lbl_h = tk.Label(lf2, text="Target Wall Build Height (H_build):", font=FONT_LABEL)
        lbl_h.grid(row=5, column=0, sticky=tk.W, padx=(8, 4), pady=2)
        self._reg(lbl_h, "label")

        self._build_height_var = tk.StringVar(value=str(round(self._current_build_height_mm(), 2)))
        self._build_height_entry = tk.Entry(lf2, textvariable=self._build_height_var, width=10, font=FONT_ENTRY, relief=tk.SOLID, bd=1)
        self._build_height_entry.grid(row=5, column=1, padx=4, pady=2, sticky=tk.W)
        self._reg(self._build_height_entry, "entry")
        self._build_height_entry.bind("<FocusIn>", lambda e: self._on_build_height_focus())
        self._build_height_entry.bind("<KeyRelease>", lambda e: self._on_build_height_key())

        u_lbl = tk.Label(lf2, text="mm", font=FONT_SMALL)
        u_lbl.grid(row=5, column=2, sticky=tk.W, padx=4)
        self._reg(u_lbl, "label_dim")

        self._build_height_var.trace_add("write", lambda *a: self._on_build_height_change())

        # Row 6: Deposition Pattern Selection (Torch Motion Strategy)
        pattern_frame = tk.Frame(lf2)
        pattern_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=8, pady=(4, 4))
        self._reg(pattern_frame, "card_frame")

        lbl_pat = tk.Label(pattern_frame, text="Deposition Pattern:", font=FONT_SMALL_BOLD)
        lbl_pat.pack(side=tk.LEFT, padx=(0, 8))
        self._reg(lbl_pat, "label")

        self.deposition_pattern_var = tk.StringVar(value="bidirectional")
        rb_bi = tk.Radiobutton(
            pattern_frame, text="Bidirectional (Zigzag / Alternating Direction)",
            variable=self.deposition_pattern_var, value="bidirectional",
            font=FONT_SMALL, command=self._on_deposition_pattern_change
        )
        rb_bi.pack(side=tk.LEFT, padx=6)
        self._reg(rb_bi, "radio")

        rb_uni = tk.Radiobutton(
            pattern_frame, text="Unidirectional (Same Direction Every Pass)",
            variable=self.deposition_pattern_var, value="unidirectional",
            font=FONT_SMALL, command=self._on_deposition_pattern_change
        )
        rb_uni.pack(side=tk.LEFT, padx=6)
        self._reg(rb_uni, "radio")

        # Row 7: Live Fusion Profile Badge
        self.geom_fusion_info_lbl = tk.Label(
            lf2, text="—", font=FONT_SMALL_BOLD, fg="#2563eb", justify=tk.LEFT, anchor=tk.W
        )
        self.geom_fusion_info_lbl.grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=8, pady=(4, 6))
        self._reg(self.geom_fusion_info_lbl, "label")
        self._update_geom_fusion_label()
        self._on_layer_mode_change()
        self._auto_update_wall_gcode()

    # -- numeric helpers ---------------------------------------------------
    @staticmethod
    def _f(var_name, default):
        try:
            raw = var_name.get().strip()
            return float(raw) if raw else float(default)
        except Exception:
            return float(default)

    def _current_build_height_mm(self):
        h1 = self._f(self._param_vars["h_bead"][0], 4.0) if "h_bead" in self._param_vars else 4.0
        remelt = (self._f(self._param_vars["remelt_ratio"][0], 0.8)
                  if "remelt_ratio" in self._param_vars else 0.8)
        nl = 1
        if "n_layers" in self._param_vars:
            try:
                nl = max(int(float(self._param_vars["n_layers"][0].get().strip() or "1")), 1)
            except Exception:
                nl = 1
        return h1 if nl <= 1 else (h1 + (nl - 1) * h1 * remelt)

    def _on_n_layers_focus(self):
        if hasattr(self, "_layer_mode") and self._layer_mode.get() != "n_layers":
            self._layer_mode.set("n_layers")
            self._on_layer_mode_change()

    def _on_n_layers_key(self):
        if hasattr(self, "_layer_mode") and self._layer_mode.get() != "n_layers":
            self._layer_mode.set("n_layers")
        self._sync_from_n_layers()
        self._update_geom_fusion_label()
        self._update_mesh_calc()
        self._update_heat_calc()
        self._auto_sync_probe_points()

    def _on_build_height_focus(self):
        if hasattr(self, "_layer_mode") and self._layer_mode.get() != "height":
            self._layer_mode.set("height")
            self._on_layer_mode_change()

    def _on_build_height_key(self):
        if hasattr(self, "_layer_mode") and self._layer_mode.get() != "height":
            self._layer_mode.set("height")
        self._sync_from_h_bead()
        self._update_geom_fusion_label()
        self._update_mesh_calc()
        self._update_heat_calc()
        self._auto_sync_probe_points()

    def _sync_from_n_layers(self):
        """Total layer count N_layer drives the build: compute the resulting total wall height."""
        if not hasattr(self, "_param_vars") or "n_layers" not in self._param_vars:
            return
        if getattr(self, "_syncing_layer_mode", False):
            return
        self._syncing_layer_mode = True
        try:
            h1 = self._f(self._param_vars["h_bead"][0], 4.0) if "h_bead" in self._param_vars else 4.0
            remelt = (self._f(self._param_vars["remelt_ratio"][0], 0.8)
                      if "remelt_ratio" in self._param_vars else 0.8)
            try:
                nl = max(int(float(self._param_vars["n_layers"][0].get().strip() or "1")), 1)
            except Exception:
                return
            h_sub = max(h1 * remelt, 1e-4)
            total_h = h1 if nl <= 1 else (h1 + (nl - 1) * h_sub)
            if hasattr(self, "_build_height_var"):
                cur_h = self._build_height_var.get().strip()
                new_h = f"{total_h:.2f}"
                if cur_h != new_h:
                    self._build_height_var.set(new_h)
        finally:
            self._syncing_layer_mode = False

    def _sync_from_h_bead(self):
        """Target build height H_build drives the build: compute the required N_layer."""
        if not hasattr(self, "_build_height_var") or not hasattr(self, "_param_vars"):
            return
        if getattr(self, "_syncing_layer_mode", False):
            return
        self._syncing_layer_mode = True
        try:
            h1 = self._f(self._param_vars["h_bead"][0], 4.0) if "h_bead" in self._param_vars else 4.0
            remelt = (self._f(self._param_vars["remelt_ratio"][0], 0.8)
                      if "remelt_ratio" in self._param_vars else 0.8)
            target_h = self._f(self._build_height_var, 0.0)
            if target_h <= 0 or h1 <= 0:
                return
            h_sub = max(h1 * remelt, 1e-4)
            if target_h <= h1:
                nl = 1
            else:
                nl = 1 + int(math.ceil((target_h - h1) / h_sub - 1e-6))
            nl = max(1, nl)
            if "n_layers" in self._param_vars:
                cur_nl = self._param_vars["n_layers"][0].get().strip()
                if cur_nl != str(nl):
                    self._param_vars["n_layers"][0].set(str(nl))
        finally:
            self._syncing_layer_mode = False

    def _effective_layer_count(self):
        """Single source of truth for N_layer under the active driver.

        Returns (nl, target_h_mm_or_None, actual_h_mm). In height-driver
        mode the layer count is derived from H_build with a ceil (the same
        formula _collect_params uses for the solver), so the label, the
        entry and the EDP can never disagree.
        """
        try:
            h1 = self._f(self._param_vars["h_bead"][0], 4.0) if "h_bead" in self._param_vars else 4.0
        except Exception:
            h1 = 4.0
        try:
            remelt = self._f(self._param_vars["remelt_ratio"][0], 0.8) if "remelt_ratio" in self._param_vars else 0.8
        except Exception:
            remelt = 0.8
        try:
            raw_nl = self._param_vars["n_layers"][0].get().strip() if "n_layers" in self._param_vars else ""
            nl_entry = max(int(float(raw_nl)), 1) if raw_nl else 1
        except Exception:
            nl_entry = 1
        mode = getattr(self, "_layer_mode", None)
        mode_val = mode.get() if mode is not None else "n_layers"
        h_sub = max(h1 * remelt, 1e-4)
        if mode_val in ("height", "h_bead"):
            try:
                target_h = self._f(getattr(self, "_build_height_var", None), 0.0)
            except Exception:
                target_h = 0.0
            if target_h > 0 and h1 > 0:
                nl = 1 if target_h <= h1 else 1 + int(math.ceil((target_h - h1) / h_sub - 1e-6))
                nl = max(1, nl)
                actual_h = h1 if nl <= 1 else (h1 + (nl - 1) * h_sub)
                return nl, target_h, actual_h
            actual_h = h1 if nl_entry <= 1 else (h1 + (nl_entry - 1) * h_sub)
            return nl_entry, None, actual_h
        actual_h = h1 if nl_entry <= 1 else (h1 + (nl_entry - 1) * h_sub)
        return nl_entry, None, actual_h

    def _on_layer_mode_change(self):
        try:
            self._mark_custom_preset()
        except Exception:
            pass
        mode = getattr(self, "_layer_mode", None)
        if mode is None:
            return
        is_n = (mode.get() == "n_layers")
        # Both fields remain editable at all times; syncing aligns them
        if is_n:
            self._sync_from_n_layers()
        else:
            self._sync_from_h_bead()
        self._update_geom_fusion_label()
        self._update_mesh_calc()
        self._update_heat_calc()
        self._auto_sync_probe_points()

    def _on_build_height_change(self):
        try:
            self._mark_custom_preset()
        except Exception:
            pass
        if getattr(self, "_syncing_layer_mode", False):
            return
        mode = getattr(self, "_layer_mode", None)
        if mode is None or mode.get() not in ("h_bead", "height"):
            return
        self._sync_from_h_bead()
        self._update_geom_fusion_label()
        self._update_mesh_calc()
        self._update_heat_calc()
        self._auto_sync_probe_points()

    def _auto_sync_probe_points(self):
        """Auto-update probe points and paths in Solver tab from Geometry window unless user custom coordinates active."""
        if hasattr(self, "probe_points_text") and not getattr(self, "_user_custom_probes", False):
            try:
                default_txt = self._get_default_probe_points_text()
                self.probe_points_text.delete("1.0", tk.END)
                self.probe_points_text.insert("1.0", default_txt)
            except Exception:
                pass
        if hasattr(self, "probe_paths_text") and not getattr(self, "_user_custom_paths", False):
            try:
                path_txt = self._get_default_probe_paths_text()
                self.probe_paths_text.delete("1.0", tk.END)
                self.probe_paths_text.insert("1.0", path_txt)
            except Exception:
                pass
        self._update_probe_geom_badge()

    def _update_geom_fusion_label(self):
        """Update dynamic info label in Geometry tab for Layer 1, subsequent layers, total build height, and pattern."""
        if not hasattr(self, "geom_fusion_info_lbl"):
            return
        try:
            h1 = 4.0
            if "h_bead" in self._param_vars:
                raw = self._param_vars["h_bead"][0].get().strip()
                if raw:
                    h1 = float(raw)
            remelt = 0.80
            if "remelt_ratio" in self._param_vars:
                raw_r = self._param_vars["remelt_ratio"][0].get().strip()
                if raw_r:
                    remelt = float(raw_r)
            nl_eff, target_h, h_total = self._effective_layer_count()
            nl = nl_eff
            h_sub = h1 * remelt
            pat_str = self.deposition_pattern_var.get().capitalize() if hasattr(self, "deposition_pattern_var") else "Bidirectional"
            mode = getattr(self, "_layer_mode", None)
            mode_val = mode.get() if mode is not None else "n_layers"
            if mode_val in ("height", "h_bead") and target_h:
                txt = (
                    f"ℹ Fusion Profile: Layer 1 = {h1:.2f} mm | Subsequent = {h_sub:.2f} mm "
                    f"({remelt*100.0:.0f}% height retention) | Driver: Target H_build = {target_h:.2f} mm → "
                    f"N_layer = {nl} | Actual Wall Height = {h_total:.2f} mm | Pattern: {pat_str}"
                )
            else:
                txt = (
                    f"ℹ Fusion Profile: Layer 1 = {h1:.2f} mm | Subsequent = {h_sub:.2f} mm "
                    f"({remelt*100.0:.0f}% height retention) | Driver: N_layer = {nl} | "
                    f"Total Wall Height = {h_total:.2f} mm | Pattern: {pat_str}"
                )
            self.geom_fusion_info_lbl.config(text=txt)
        except Exception:
            pass

    def _on_deposition_pattern_change(self):
        """Callback when user toggles between Unidirectional and Bidirectional pattern."""
        try:
            self._mark_custom_preset()
        except Exception:
            pass
        self._update_geom_fusion_label()
        self._auto_update_wall_gcode()

    def _auto_update_wall_gcode(self, run_dir=None):
        """
        Dynamically synthesize and auto-update toolpath G-code (toolpath.gcode),
        FreeFEM++ toolpath reader array (toolpath.txt), and per-layer pass times (layer_pass_times.txt)
        for the active straight wall geometry and chosen deposition pattern.
        """
        try:
            # 1. Retrieve active geometry & kinematics parameters
            lx_mm = self._f(self._param_vars.get("lx", (None,))[0], 150.0) if hasattr(self, "_param_vars") and "lx" in self._param_vars else 150.0
            ly_base_mm = self._f(self._param_vars.get("ly_base", (None,))[0], 150.0) if hasattr(self, "_param_vars") and "ly_base" in self._param_vars else 150.0
            hz_base_mm = self._f(self._param_vars.get("hz_base", (None,))[0], 10.0) if hasattr(self, "_param_vars") and "hz_base" in self._param_vars else 10.0
            w_bead_mm = self._f(self._param_vars.get("ty_dep", (None,))[0], 5.0) if hasattr(self, "_param_vars") and "ty_dep" in self._param_vars else 5.0
            h1_mm = self._f(self._param_vars.get("h_bead", (None,))[0], 4.0) if hasattr(self, "_param_vars") and "h_bead" in self._param_vars else 4.0
            remelt = self._f(self._param_vars.get("remelt_ratio", (None,))[0], 0.8) if hasattr(self, "_param_vars") and "remelt_ratio" in self._param_vars else 0.8
            
            nl = 10
            if hasattr(self, "_param_vars") and "n_layers" in self._param_vars:
                try:
                    nl = max(int(float(self._param_vars["n_layers"][0].get().strip() or "1")), 1)
                except Exception:
                    nl = 10

            v_scan_mms = self._f(self._param_vars.get("v_scan", (None,))[0], 10.0) if hasattr(self, "_param_vars") and "v_scan" in self._param_vars else 10.0
            if v_scan_mms < 0.05:
                v_scan_mms *= 1000.0  # convert m/s to mm/s if in metres
            v_scan_mms = max(v_scan_mms, 0.1)
            feedrate_mmpm = v_scan_mms * 60.0
            rapid_mmpm = max(feedrate_mmpm * 5.0, 3000.0)

            t_dwell = self._f(self._param_vars.get("t_wait", (None,))[0], 10.0) if hasattr(self, "_param_vars") and "t_wait" in self._param_vars else 10.0
            i_weld = self._f(self._param_vars.get("I_weld", (None,))[0], 200.0) if hasattr(self, "_param_vars") and "I_weld" in self._param_vars else 200.0
            u_weld = self._f(self._param_vars.get("U_weld", (None,))[0], 24.0) if hasattr(self, "_param_vars") and "U_weld" in self._param_vars else 24.0
            power_w = i_weld * u_weld

            pattern = self.deposition_pattern_var.get() if hasattr(self, "deposition_pattern_var") else "bidirectional"
            is_bidi = (pattern.lower() == "bidirectional")

            # 2. Determine target directories
            target_dirs = []
            if run_dir:
                target_dirs.append(run_dir)
                fem_sub = os.path.join(run_dir, "FreeFEM")
                if os.path.exists(fem_sub):
                    target_dirs.append(fem_sub)
            else:
                # Active case or default case directory
                cur_out = self._get_current_run_dir() if hasattr(self, "_get_current_run_dir") else None
                if cur_out and os.path.exists(cur_out):
                    target_dirs.append(cur_out)
                    fem_sub = os.path.join(cur_out, "FreeFEM")
                    if os.path.exists(fem_sub):
                        target_dirs.append(fem_sub)
                if hasattr(self, "case_dir") and self.case_dir and os.path.exists(self.case_dir):
                    target_dirs.append(self.case_dir)

            if not target_dirs:
                base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "waam_output")
                os.makedirs(base_dir, exist_ok=True)
                target_dirs.append(base_dir)

            # 3. Synthesize G-code, toolpath.txt, and layer_pass_times.txt
            y_center_mm = ly_base_mm / 2.0
            x_min_mm = 0.0
            x_max_mm = lx_mm

            gcode_lines = [
                "; ============================================================",
                "; WAAM PRODUCTION TOOLPATH — AUTO-GENERATED BY WAAM-FRAMEWORK",
                "; ============================================================",
                f"; Deposition Pattern:       {pattern.upper()}",
                f"; Substrate Dimensions:     {lx_mm:.1f} x {ly_base_mm:.1f} x {hz_base_mm:.1f} mm",
                f"; Number of Layers:         {nl}",
                f"; Layer 1 Height:           {h1_mm:.2f} mm",
                f"; Subsequent Bead Height:   {h1_mm * remelt:.2f} mm (remelt={remelt*100:.0f}%)",
                f"; Bead Width:               {w_bead_mm:.2f} mm",
                f"; Torch Travel Speed:       {v_scan_mms:.1f} mm/s ({feedrate_mmpm:.0f} mm/min)",
                f"; Inter-pass Dwell Time:    {t_dwell:.1f} s",
                f"; Welding Power:            {power_w:.0f} W",
                "; ============================================================",
                "G90 ; Absolute Coordinates",
                "G21 ; Millimeter Units",
                "M82 ; Absolute Extruder / Wire Feed",
                f"M3 S{power_w:.0f} ; Arc ON",
                ""
            ]

            txt_lines = []
            layer_pass_times = []
            t_curr = 0.0

            z_curr_mm = hz_base_mm + h1_mm
            txt_lines.append(f"{t_curr:.3f} 0.00000 {y_center_mm/1000.0:.5f} {z_curr_mm/1000.0:.5f} 0")

            n_substeps = max(10, int(lx_mm / 2.0))

            for layer_idx in range(1, nl + 1):
                if layer_idx == 1:
                    z_curr_mm = hz_base_mm + h1_mm
                else:
                    z_curr_mm += h1_mm * remelt

                forward = True if (not is_bidi or layer_idx % 2 == 1) else False
                x_start = x_min_mm if forward else x_max_mm
                x_end = x_max_mm if forward else x_min_mm

                gcode_lines.append(f"; --- LAYER {layer_idx} (Z = {z_curr_mm:.3f} mm, {'FWD' if forward else 'REV'}) ---")
                
                if not forward or (not is_bidi and layer_idx > 1):
                    gcode_lines.append(f"G0 X{x_start:.3f} Y{y_center_mm:.3f} Z{z_curr_mm:.3f} F{rapid_mmpm:.0f}")
                    rapid_dist = abs(x_max_mm - x_min_mm)
                    dt_rapid = rapid_dist / (rapid_mmpm / 60.0)
                    t_curr += dt_rapid
                    txt_lines.append(f"{t_curr:.3f} {x_start/1000.0:.5f} {y_center_mm/1000.0:.5f} {z_curr_mm/1000.0:.5f} 0")
                else:
                    gcode_lines.append(f"G0 Z{z_curr_mm:.3f} F{rapid_mmpm:.0f}")

                gcode_lines.append(f"G1 X{x_end:.3f} Y{y_center_mm:.3f} F{feedrate_mmpm:.0f}")

                t_pass = abs(x_end - x_start) / v_scan_mms
                layer_pass_times.append(t_pass)

                for step in range(1, n_substeps + 1):
                    frac = step / float(n_substeps)
                    x_sub = x_start + frac * (x_end - x_start)
                    t_sub = t_curr + frac * t_pass
                    txt_lines.append(f"{t_sub:.3f} {x_sub/1000.0:.5f} {y_center_mm/1000.0:.5f} {z_curr_mm/1000.0:.5f} 1")

                t_curr += t_pass

                if t_dwell > 0:
                    gcode_lines.append(f"G4 P{int(t_dwell * 1000)} ; Inter-pass dwell ({t_dwell:.1f}s)")
                    t_curr += t_dwell
                    txt_lines.append(f"{t_curr:.3f} {x_end/1000.0:.5f} {y_center_mm/1000.0:.5f} {z_curr_mm/1000.0:.5f} 0")
                
                gcode_lines.append("")

            gcode_lines.append("M5 ; Arc OFF")
            gcode_lines.append(f"G0 Z{z_curr_mm + 25.0:.3f} F{rapid_mmpm:.0f} ; Safe Retract")
            gcode_lines.append("; End of WAAM G-code")

            gcode_text = "\n".join(gcode_lines) + "\n"
            tp_text = f"{len(txt_lines)}\n" + "\n".join(txt_lines) + "\n"
            lt_lines = [f"{len(layer_pass_times)}"] + [f"{i+1} {pt:.6f}" for i, pt in enumerate(layer_pass_times)]
            lt_text = "\n".join(lt_lines) + "\n"

            for d in set(target_dirs):
                try:
                    os.makedirs(d, exist_ok=True)
                    with open(os.path.join(d, "toolpath.gcode"), "w", encoding="utf-8") as f:
                        f.write(gcode_text)
                    with open(os.path.join(d, "toolpath.txt"), "w", encoding="utf-8") as f:
                        f.write(tp_text)
                    with open(os.path.join(d, "layer_pass_times.txt"), "w", encoding="utf-8") as f:
                        f.write(lt_text)
                except Exception:
                    pass

            return True
        except Exception as e:
            print(f"[GUI] Auto-update wall G-code warning: {e}")
            return False

    def _build_process_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Process")
        self._reg(tab, "card_frame")

        lf1 = tk.LabelFrame(tab, text="Arc / Energy Process Parameters", font=FONT_SECTION, padx=8, pady=4)
        lf1.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf1, "card_labelframe")
        self._add_info_btn_to_labelframe(lf1, self._show_process_info)
        self._make_param_row(lf1, "Arc Current (I_arc):", "I_weld", 0, "A")
        self._make_param_row(lf1, "Arc Voltage (U_arc):", "U_weld", 1, "V")
        self._make_param_row(lf1, "Thermal Efficiency (η_arc):", "eta_arc", 2, "[-]")
        self._make_param_row(lf1, "Torch Travel Speed (v_scan / v_travel):", "v_scan", 3, "mm/s")
        self._make_param_row(lf1, "Wire Feed Speed (WFS / v_wire):", "v_wire", 4, "m/min")
        self._make_param_row(lf1, "Wire Diameter (d_wire):", "d_wire", 5, "mm")
        self._make_param_row(lf1, "Inter-Pass Dwell (t_dwell):", "t_wait", 6, "s")

        # Calculated Heat Input & Energy Output Card
        self.heat_calc_frame = tk.LabelFrame(
            tab, text="Calculated Heat Input & Energy Output",
            font=FONT_SECTION, padx=8, pady=4
        )
        self.heat_calc_frame.pack(fill=tk.X, padx=8, pady=3)
        self._reg(self.heat_calc_frame, "card_labelframe")
        self._add_info_btn_to_labelframe(self.heat_calc_frame, self._show_heat_calc_info)

        self.heat_calc_label = tk.Label(
            self.heat_calc_frame, text="—",
            font=FONT_MONO, justify=tk.LEFT, anchor=tk.W
        )
        self.heat_calc_label.pack(fill=tk.X, padx=4, pady=2)
        self._reg(self.heat_calc_label, "label")

        lf2 = tk.LabelFrame(tab, text="Heat Source Model", font=FONT_SECTION, padx=8, pady=4)
        lf2.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf2, "card_labelframe")
        self._add_info_btn_to_labelframe(lf2, self._show_heatsource_info)

        sources = [
            ("goldak", "Goldak Double Ellipsoid (Arc / GMAW WAAM)"),
            ("p48_surface", "P48 Surface Arc Flux + Droplet Impingement (GMAW ER70S6 / 316L Thin-Wall)"),
            ("gaussian_vol", "3D Gaussian Volumetric (Laser-DED)"),
            ("gaussian_surf", "2D Gaussian Surface Heat Flux (Thin powder/wire)"),
            ("conical", "Conical Frustum (Keyhole / Plasma PTA)"),
            ("uniform", "Uniform Moving Band (Wide weave / Brazing)"),
            ("prescribed", "Prescribed Liquidus Temperature (Fast)"),
        ]

        self.source_var = tk.StringVar(value=self.params["heat_source"])
        for i, (val, text) in enumerate(sources):
            rb = tk.Radiobutton(
                lf2, text=text, variable=self.source_var, value=val,
                font=FONT_SMALL, command=self._update_source_params
            )
            rb.grid(row=i, column=0, sticky=tk.W, padx=4, pady=1)
            self._reg(rb, "radio")

        self.source_params_frame = tk.LabelFrame(tab, text="Active Heat Source Dimensions", font=FONT_SECTION, padx=8, pady=4)
        self.source_params_frame.pack(fill=tk.X, padx=8, pady=3)
        self._reg(self.source_params_frame, "card_labelframe")
        self._add_info_btn_to_labelframe(self.source_params_frame, self._show_heatsource_info)
        self._update_source_params()

    def _update_source_params(self):
        for w in self.source_params_frame.winfo_children():
            w.destroy()
        src = self.source_var.get()
        if src == "goldak":
            self._make_param_row(self.source_params_frame, "Front Semi-Axis (a_f):", "af", 0, "mm")
            self._make_param_row(self.source_params_frame, "Rear Semi-Axis (a_r):", "ar", 1, "mm")
            self._make_param_row(self.source_params_frame, "Melt Half-Width (b_G):", "bG", 2, "mm")
            self._make_param_row(self.source_params_frame, "Penetration Depth (c_G):", "cG", 3, "mm")
            self._make_param_row(self.source_params_frame, "Front Power Fraction (f_f):", "ff", 4, "[-]")
            self._make_param_row(self.source_params_frame, "Rear Power Fraction (f_r):", "fr", 5, "[-]")
        elif src in ("p48_surface", "p48"):
            self._make_param_row(self.source_params_frame, "Droplet Enthalpy Fraction (η_d):", "eta_drop", 0, "[-]")
            self._make_param_row(self.source_params_frame, "Droplet Transfer Temp (T_drop):", "T_drop", 1, "°C")
            try:
                i_w = float(self._param_vars["I_weld"][0].get().strip() or "150")
            except Exception:
                i_w = 150.0
            try:
                eta_d = float(self._param_vars.get("eta_drop", [None])[0].get().strip() or "0.25")
            except Exception:
                eta_d = 0.25
            sig_q = 6.205e-4 * (max(i_w, 10.0) ** 0.28) * 1000.0
            lbl_info = tk.Label(
                self.source_params_frame,
                text=f"ⓘ P48 Arc Radius: σ_q(I) = 6.205×10⁻⁴·I⁰·²⁸ = {sig_q:.2f} mm | Direct Flux = {(1.0-eta_d)*100:.0f}% | Droplet = {eta_d*100:.0f}%",
                font=FONT_SMALL_BOLD, fg="#0284c7"
            )
            lbl_info.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=8, pady=(4, 2))
            self._reg(lbl_info, "label")
        elif src == "gaussian_vol":
            self._make_param_row(self.source_params_frame, "Beam Radius (R_beam):", "R_beam", 0, "mm")
        elif src == "gaussian_surf":
            self._make_param_row(self.source_params_frame, "Surface Spot Radius (R_spot):", "R_spot", 0, "mm")
        elif src == "conical":
            self._make_param_row(self.source_params_frame, "Top Radius (r_top):", "r_top", 0, "mm")
            self._make_param_row(self.source_params_frame, "Bottom Radius (r_bot):", "r_bot", 1, "mm")
            self._make_param_row(self.source_params_frame, "Penetration Depth (d_cone):", "depth_cone", 2, "mm")
        elif src == "uniform":
            self._make_param_row(self.source_params_frame, "Half-Length (X):", "a_band", 0, "mm")
            self._make_param_row(self.source_params_frame, "Half-Width (Y):", "b_band", 1, "mm")
            self._make_param_row(self.source_params_frame, "Depth (Z):", "c_band", 2, "mm")
        elif src == "prescribed":
            self._make_param_row(self.source_params_frame, "Activation Radius:", "R_activate", 0, "mm")
        self._apply_theme()

    def _build_material_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Material")
        self._reg(tab, "card_frame")

        mat_names = list(self.materials.keys()) if self.materials else ["SS308L", "SS316L", "Ti-6Al-4V", "Pure Al (Al1xxx)"]

        # 1. Dual Material Assignment (Substrate vs Deposit)
        lf1 = tk.LabelFrame(tab, text="Alloy Assignment (Dissimilar Multi-Material)", font=FONT_SECTION, padx=8, pady=4)
        lf1.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf1, "card_labelframe")
        self._add_info_btn_to_labelframe(lf1, self._show_material_info)

        lbl_base = tk.Label(lf1, text="Substrate Alloy (Base Plate):", font=FONT_SMALL_BOLD)
        lbl_base.grid(row=0, column=0, padx=6, pady=3, sticky=tk.W)
        self._reg(lbl_base, "label")

        self.material_base_var = tk.StringVar(value=self.params.get("material_base", "SS316L"))
        base_combo = ttk.Combobox(lf1, textvariable=self.material_base_var, values=mat_names, width=22, state="readonly", font=FONT_SMALL)
        base_combo.grid(row=0, column=1, padx=4, pady=3, sticky=tk.W)
        base_combo.bind("<<ComboboxSelected>>", self._on_multi_material_change)

        lbl_dep = tk.Label(lf1, text="Deposit Bead Alloy (Wire):", font=FONT_SMALL_BOLD)
        lbl_dep.grid(row=1, column=0, padx=6, pady=3, sticky=tk.W)
        self._reg(lbl_dep, "label")

        self.material_dep_var = tk.StringVar(value=self.params.get("material_dep", "SS308L"))
        dep_combo = ttk.Combobox(lf1, textvariable=self.material_dep_var, values=mat_names, width=22, state="readonly", font=FONT_SMALL)
        dep_combo.grid(row=1, column=1, padx=4, pady=3, sticky=tk.W)
        dep_combo.bind("<<ComboboxSelected>>", self._on_multi_material_change)

        lbl_wire_dia = tk.Label(lf1, text="Wire Diameter / Size (d_wire):", font=FONT_SMALL_BOLD)
        lbl_wire_dia.grid(row=2, column=0, padx=6, pady=3, sticky=tk.W)
        self._reg(lbl_wire_dia, "label")

        wire_var = self._param_vars["d_wire"][0] if "d_wire" in self._param_vars else tk.StringVar(value="1.2")
        self.material_wire_combo = ttk.Combobox(
            lf1, textvariable=wire_var,
            values=["0.8", "0.9", "1.0", "1.2", "1.6", "2.0", "2.4", "3.2"],
            width=10, font=FONT_SMALL
        )
        self.material_wire_combo.grid(row=2, column=1, padx=4, pady=3, sticky=tk.W)
        self.material_wire_combo.bind("<<ComboboxSelected>>", lambda e: self._on_any_param_change("d_wire"))

        lbl_wire_u = tk.Label(lf1, text="mm (AWS/ISO Solid Wire Feedstock)", font=FONT_SMALL)
        lbl_wire_u.grid(row=2, column=2, padx=4, pady=3, sticky=tk.W)
        self._reg(lbl_wire_u, "label_dim")

        self.material_feed_info = tk.Label(
            lf1, text="—", font=FONT_MONO, justify=tk.LEFT, anchor=tk.W,
            relief=tk.SOLID, bd=1, padx=6, pady=4
        )
        self.material_feed_info.grid(row=3, column=0, columnspan=3, padx=6, pady=4, sticky=tk.EW)
        self._reg(self.material_feed_info, "entry")

        # 2. Dissimilar Metal Interface Mismatch & Thermophysical Compatibility Card
        lf_mismatch = tk.LabelFrame(tab, text="Interfacial Thermophysical & CTE Compatibility", font=FONT_SECTION, padx=8, pady=4)
        lf_mismatch.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_mismatch, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_mismatch, self._show_material_info)

        self.dissimilar_info = tk.Text(lf_mismatch, height=5, width=60, font=FONT_MONO, state=tk.DISABLED, relief=tk.SOLID, bd=1)
        self.dissimilar_info.pack(fill=tk.X, padx=4, pady=4)
        self._reg(self.dissimilar_info, "entry")

        # 3. Physical Constants (Override)
        lf2 = tk.LabelFrame(tab, text="Physical Constants & Thermal Boundary Loss", font=FONT_SECTION, padx=8, pady=4)
        lf2.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf2, "card_labelframe")
        self._add_info_btn_to_labelframe(lf2, self._show_properties_info)
        self._make_param_row(lf2, "Deposit Density (ρ):", "rho", 0, "kg/m³")
        self._make_param_row(lf2, "Solidus Temp (T_sol):", "T_sol", 1, "°C")
        self._make_param_row(lf2, "Liquidus Temp (T_liq):", "T_liq", 2, "°C")
        self._make_param_row(lf2, "Latent Heat of Fusion (L_f):", "latent_heat", 3, "J/kg")
        self._make_param_row(lf2, "Surface Emissivity (ε):", "emissivity", 4, "[-]")
        self._make_param_row(lf2, "Convection Coeff (h_conv):", "h_conv", 5, "W/(m²·K)")
        self._make_param_row(lf2, "Base Contact Loss (h_bot):", "h_bot", 6, "W/(m²·K)")
        self._make_param_row(lf2, "Mechanical Cutoff (T_mech):", "T_mech", 7, "°C")

        self._on_multi_material_change()

    def _update_material_wire_readout(self):
        """Update live wire feedstock cross section, mass deposition, and bead continuity readout."""
        if not hasattr(self, "material_feed_info"):
            return
        try:
            dep_key = self.material_dep_var.get() if hasattr(self, "material_dep_var") else "SS308L"
            mat_dep = self.materials.get(dep_key, {})
            rho_val = float(mat_dep.get("density", 7850.0))
            if "rho" in self._param_vars:
                try:
                    rho_val = float(self._param_vars["rho"][0].get())
                except Exception:
                    pass

            d_w_val = float(self._param_vars["d_wire"][0].get()) if "d_wire" in self._param_vars else 1.2
            d_w_mm = d_w_val * 1000.0 if d_w_val < 0.05 else d_w_val

            wfs_val = float(self._param_vars["v_wire"][0].get()) if "v_wire" in self._param_vars else 5.0
            v_scan_val = float(self._param_vars["v_scan"][0].get()) if "v_scan" in self._param_vars else 8.0
            v_scan_mms = v_scan_val * 1000.0 if v_scan_val < 0.05 else v_scan_val

            a_wire_mm2 = (math.pi * (d_w_mm ** 2)) / 4.0
            q_wire_mm3_min = a_wire_mm2 * (wfs_val * 1000.0)
            q_wire_cm3_s = q_wire_mm3_min / 60000.0
            m_dep_kg_h = (q_wire_mm3_min * (rho_val * 1e-9)) * 60.0
            m_dep_g_s = (m_dep_kg_h * 1000.0) / 3600.0

            v_travel_mm_min = v_scan_mms * 60.0
            a_bead_theo_mm2 = (q_wire_mm3_min / v_travel_mm_min) if v_travel_mm_min > 0 else 0.0

            w_bead = float(self._param_vars["ty_dep"][0].get()) if "ty_dep" in self._param_vars else 8.0
            w_bead_mm = w_bead * 1000.0 if w_bead < 0.05 else w_bead
            h_bead = float(self._param_vars["h_bead"][0].get()) if "h_bead" in self._param_vars else 3.0
            h_bead_mm = h_bead * 1000.0 if h_bead < 0.05 else h_bead
            a_bead_cad_mm2 = w_bead_mm * h_bead_mm

            txt = (
                f"• Feedstock Wire:  ∅ {d_w_mm:.2f} mm | Aw = {a_wire_mm2:.3f} mm² | Vol Feed: {q_wire_cm3_s:.3f} cm³/s\n"
                f"• Mass Deposition: ṁ = {m_dep_kg_h:.2f} kg/h ({m_dep_g_s:.2f} g/s) [{dep_key}, ρ={rho_val:.0f} kg/m³]\n"
                f"• Bead Continuity: A_theo = {a_bead_theo_mm2:.2f} mm² vs CAD = {a_bead_cad_mm2:.2f} mm² (w={w_bead_mm:.1f}mm, h={h_bead_mm:.1f}mm)"
            )
            self.material_feed_info.config(text=txt)
        except Exception:
            pass

    def _on_multi_material_change(self, event=None):
        base_key = self.material_base_var.get()
        dep_key = self.material_dep_var.get()

        mat_base = self.materials.get(base_key, {})
        mat_dep = self.materials.get(dep_key, {})

        def get_cte(m):
            te = m.get("thermal_expansion", {}).get("alpha", [1.2e-5])
            return te[0] if te else 1.2e-5

        def get_k(m):
            cd = m.get("conductivity", {}).get("k", [30.0])
            return cd[0] if cd else 30.0

        alpha_b = get_cte(mat_base)
        alpha_d = get_cte(mat_dep)
        delta_alpha = (alpha_d - alpha_b) * 1e6

        k_b = get_k(mat_base)
        k_d = get_k(mat_dep)

        tliq_b = mat_base.get("liquidus", 1520)
        tliq_d = mat_dep.get("liquidus", 1440)
        delta_tliq = tliq_d - tliq_b

        is_dissimilar = (base_key != dep_key)
        if is_dissimilar:
            if abs(delta_alpha) >= 4.0:
                risk_status = f"[WARN] HIGH CTE MISMATCH (Δα = {delta_alpha:+.2f}×10⁻⁶/K)\n   Severe interfacial shear & delamination risk upon cooling!"
            elif abs(delta_alpha) >= 2.0:
                risk_status = f"MODERATE CTE MISMATCH (Δα = {delta_alpha:+.2f}×10⁻⁶/K)\n   Interface thermal stress will produce pronounced bimetallic bending."
            else:
                risk_status = f"[OK] LOW CTE MISMATCH (Δα = {delta_alpha:+.2f}×10⁻⁶/K)\n   Excellent thermomechanical compatibility at the fusion boundary."
        else:
            risk_status = "[OK] MONOLITHIC PRINTING (Substrate and Deposit are identical alloy)"

        info = (
            f"Substrate: {mat_base.get('name', base_key)} [α={alpha_b*1e6:.1f}×10⁻⁶/K, k={k_b:.1f}W/mK, Tliq={tliq_b}°C]\n"
            f"Deposit:   {mat_dep.get('name', dep_key)} [α={alpha_d*1e6:.1f}×10⁻⁶/K, k={k_d:.1f}W/mK, Tliq={tliq_d}°C]\n"
            f"Differential: Δα={delta_alpha:+.2f}×10⁻⁶/K | ΔTliq={delta_tliq:+.0f}°C | k_sub/k_dep={k_b/max(k_d, 0.1):.2f}\n"
            f"{risk_status}"
        )

        self.dissimilar_info.config(state=tk.NORMAL)
        self.dissimilar_info.delete("1.0", tk.END)
        self.dissimilar_info.insert("1.0", info)
        self.dissimilar_info.config(state=tk.DISABLED)

        # Update deposit overrides
        if mat_dep:
            for k in ["rho", "T_sol", "T_liq", "latent_heat", "emissivity"]:
                val = mat_dep.get("density" if k == "rho" else ("solidus" if k == "T_sol" else ("liquidus" if k == "T_liq" else k)))
                if val is not None and k in self._param_vars:
                    self._param_vars[k][0].set(str(val))
            std_wire = mat_dep.get("wire_diameter_default_mm")
            if std_wire and "d_wire" in self._param_vars:
                cur_dw = self._param_vars["d_wire"][0].get().strip()
                if cur_dw in ("", "1.2", "1.20"):
                    self._param_vars["d_wire"][0].set(str(std_wire))
        self._update_material_wire_readout()

    def _build_mesh_cad_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Mesh")
        self._reg(tab, "card_frame")

        # Scrollable container for Mesh tab
        canvas = tk.Canvas(tab, borderwidth=0, highlightthickness=0,
                           bg=self.colors["bg_card"], highlightbackground=self.colors["border"])
        self.mesh_canvas = canvas
        v_scroll = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_content = tk.Frame(canvas)
        self._reg(scroll_content, "card_frame")

        scroll_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.configure(yscrollcommand=v_scroll.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 1. Geometry & Mesh Source Mode
        lf1 = tk.LabelFrame(scroll_content, text="Geometry & Mesh Pipeline Source", font=FONT_SECTION, padx=8, pady=3)
        lf1.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf1, "card_labelframe")
        self._add_info_btn_to_labelframe(lf1, self._show_mesh_info)

        self.mesh_source_var = tk.StringVar(value="parametric")
        sources = [
            ("parametric", "Structured Hex/Tet Mesh (Parametric FreeFEM Substrate + Deposit)"),
            ("cad_file", "Import 3D CAD (.stl, .step, .stp, .iges, .igs via Gmsh 4.15 & OCCT)"),
            ("gcode_file", "Import G-code Toolpath (.gcode, .nc)"),
        ]
        for i, (val, text) in enumerate(sources):
            rb = tk.Radiobutton(
                lf1, text=text, variable=self.mesh_source_var, value=val,
                font=FONT_SMALL, command=self._update_mesh_ui
            )
            rb.grid(row=i, column=0, sticky=tk.W, padx=4, pady=1)
            self._reg(rb, "radio")

        self.file_picker_frame = tk.Frame(lf1)
        self.file_picker_frame.grid(row=3, column=0, sticky=tk.EW, padx=4, pady=3)
        self._reg(self.file_picker_frame, "card_frame")

        self.mesh_file_var = tk.StringVar()
        entry = tk.Entry(self.file_picker_frame, textvariable=self.mesh_file_var, width=32, font=FONT_MONO, relief=tk.SOLID, bd=1)
        entry.pack(side=tk.LEFT, padx=3)
        self._reg(entry, "entry")

        btn = tk.Button(self.file_picker_frame, text="Browse CAD...", command=self._browse_mesh_file, font=FONT_SMALL_BOLD, padx=4)
        btn.pack(side=tk.LEFT, padx=2)
        self._reg(btn, "card_frame")

        import_stl_btn = tk.Button(
            self.file_picker_frame, text="Import & Mesh CAD",
            command=self._import_stl, font=FONT_SMALL_BOLD,
            padx=4
        )
        import_stl_btn.pack(side=tk.LEFT, padx=2)
        self._reg(import_stl_btn, "card_frame")

        view_3d_btn = tk.Button(
            self.file_picker_frame, text="View 3D",
            command=self._open_3d_mesh_viewer, font=FONT_SMALL_BOLD,
            padx=4
        )
        view_3d_btn.pack(side=tk.LEFT, padx=2)
        self._reg(view_3d_btn, "card_frame")

        # Quick Load CAD Samples Row
        quick_frame = tk.Frame(lf1)
        quick_frame.grid(row=4, column=0, sticky=tk.W, padx=4, pady=(2, 3))
        self._reg(quick_frame, "card_frame")

        tk.Label(quick_frame, text="Quick Load CAD:", font=FONT_SMALL_BOLD).pack(side=tk.LEFT, padx=(0, 4))
        self._reg(quick_frame.winfo_children()[-1], "label")

        btn_main_stl = tk.Button(
            quick_frame, text="Load Main.stl", font=FONT_SMALL,
            command=lambda: self._load_sample_cad(str(Path(__file__).resolve().parent.parent / "Main.stl")), padx=4
        )
        btn_main_stl.pack(side=tk.LEFT, padx=2)
        self._reg(btn_main_stl, "card_frame")

        btn_k1max_step = tk.Button(
            quick_frame, text="Browse CAD File...", font=FONT_SMALL,
            command=lambda: self._browse_mesh_file(), padx=4
        )
        btn_k1max_step.pack(side=tk.LEFT, padx=2)
        self._reg(btn_k1max_step, "card_frame")

        self.cad_info_label = tk.Label(
            lf1, text="No external CAD model loaded. Using parametric geometry.",
            font=FONT_MONO, fg="#64748b", justify=tk.LEFT, anchor=tk.W
        )
        self.cad_info_label.grid(row=5, column=0, sticky=tk.W, padx=4, pady=(1, 3))
        self._reg(self.cad_info_label, "label_dim")

        # 2. Intelligent Rule-Based Auto-Mesh Engine
        lf_budget = tk.LabelFrame(scroll_content, text="Intelligent Rule-Based Auto-Mesh Engine", font=FONT_SECTION, padx=8, pady=4)
        lf_budget.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_budget, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_budget, self._show_mesh_info)

        strat_frame = tk.Frame(lf_budget)
        strat_frame.pack(fill=tk.X, pady=2)
        self._reg(strat_frame, "card_frame")

        tk.Label(strat_frame, text="Rule Strategy:", font=FONT_SMALL_BOLD).pack(side=tk.LEFT, padx=(0, 4))
        self._reg(strat_frame.winfo_children()[-1], "label")

        self.mesh_strategy_var = tk.StringVar(value="physics_adaptive")
        for s_val, s_lbl in [
            ("physics_adaptive", "Physics-Adaptive (65% HAZ Meltpool / 35% Bulk, AR <= 2.0)"),
            ("equal_aspect", "Equal-Aspect Voxel (dx ~ dy ~ dz Isotropic)"),
            ("feature_adaptive", "CAD Feature & Curvature Adaptive"),
        ]:
            rb = tk.Radiobutton(
                strat_frame, text=s_lbl, variable=self.mesh_strategy_var, value=s_val,
                font=FONT_SMALL, command=self._auto_enforce_node_budget
            )
            rb.pack(side=tk.LEFT, padx=3)
            self._reg(rb, "radio")

        budget_bar = tk.Frame(lf_budget)
        budget_bar.pack(fill=tk.X, pady=2)
        self._reg(budget_bar, "card_frame")

        tk.Label(budget_bar, text="Target Budget:", font=FONT_SMALL_BOLD).pack(side=tk.LEFT, padx=(0, 4))
        self._reg(budget_bar.winfo_children()[-1], "label")

        self.mesh_budget_var = tk.StringVar(value="2000")
        for b_val, b_lbl in [
            ("1500", "Fast (~1,500)"),
            ("2500", "Balanced (~2,500)"),
            ("5000", "High-Res (~5,000)"),
        ]:
            rb = tk.Radiobutton(
                budget_bar, text=b_lbl, variable=self.mesh_budget_var, value=b_val,
                font=FONT_SMALL, command=self._auto_enforce_node_budget
            )
            rb.pack(side=tk.LEFT, padx=3)
            self._reg(rb, "radio")

        tk.Label(budget_bar, text="Custom:", font=FONT_SMALL).pack(side=tk.LEFT, padx=(6, 2))
        self._reg(budget_bar.winfo_children()[-1], "label")
        e_custom_b = tk.Entry(budget_bar, textvariable=self.mesh_budget_var, width=6, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_custom_b.pack(side=tk.LEFT, padx=2)
        self._reg(e_custom_b, "entry")

        auto_btn = tk.Button(
            budget_bar, text="Execute Auto-Mesh", font=FONT_SMALL_BOLD,
            command=self._auto_enforce_node_budget, padx=8, pady=2
        )
        auto_btn.pack(side=tk.RIGHT, padx=3)
        self._reg(auto_btn, "card_frame")

        self.mesh_quality_label = tk.Label(
            lf_budget, text="Quality Rules: Aspect Ratio <= 2.0 | Minimum 2 cells across bead width | Dual-zone thermal grading",
            font=FONT_MONO, fg="#059669", justify=tk.LEFT, anchor=tk.W
        )
        self.mesh_quality_label.pack(anchor=tk.W, padx=2, pady=(1, 2))
        self._reg(self.mesh_quality_label, "label")

        # 3. Customizable WAAM G-Code Slicer & Toolpath Planning Card
        lf_slicer = tk.LabelFrame(scroll_content, text="Customizable WAAM G-Code Slicer & Toolpath Planning", font=FONT_SECTION, padx=8, pady=4)
        lf_slicer.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_slicer, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_slicer, self._show_mesh_info)

        # Slicer Row 1: Bead Dimensions
        s_row1 = tk.Frame(lf_slicer)
        s_row1.pack(fill=tk.X, pady=2)
        self._reg(s_row1, "card_frame")

        tk.Label(s_row1, text="Layer Height (h):", font=FONT_SMALL).pack(side=tk.LEFT, padx=2)
        self._reg(s_row1.winfo_children()[-1], "label")
        self.slicer_h_var = tk.StringVar(value="3.0")
        e_sh = tk.Entry(s_row1, textvariable=self.slicer_h_var, width=5, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_sh.pack(side=tk.LEFT, padx=2)
        self._reg(e_sh, "entry")
        tk.Label(s_row1, text="mm", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        tk.Label(s_row1, text="Bead Width (w):", font=FONT_SMALL).pack(side=tk.LEFT, padx=4)
        self._reg(s_row1.winfo_children()[-1], "label")
        self.slicer_w_var = tk.StringVar(value="8.0")
        e_sw = tk.Entry(s_row1, textvariable=self.slicer_w_var, width=5, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_sw.pack(side=tk.LEFT, padx=2)
        self._reg(e_sw, "entry")
        tk.Label(s_row1, text="mm", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        tk.Label(s_row1, text="Bead Overlap:", font=FONT_SMALL).pack(side=tk.LEFT, padx=4)
        self._reg(s_row1.winfo_children()[-1], "label")
        self.slicer_overlap_var = tk.StringVar(value="75.0")
        e_so = tk.Entry(s_row1, textvariable=self.slicer_overlap_var, width=5, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_so.pack(side=tk.LEFT, padx=2)
        self._reg(e_so, "entry")
        tk.Label(s_row1, text="%", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        tk.Label(s_row1, text="Wall Perimeters:", font=FONT_SMALL).pack(side=tk.LEFT, padx=4)
        self._reg(s_row1.winfo_children()[-1], "label")
        self.slicer_perim_var = tk.StringVar(value="1")
        e_sp = tk.Entry(s_row1, textvariable=self.slicer_perim_var, width=4, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_sp.pack(side=tk.LEFT, padx=2)
        self._reg(e_sp, "entry")
        tk.Label(s_row1, text="loops", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        # Slicer Row 2: Infill Pattern & Start Reversal
        s_row2 = tk.Frame(lf_slicer)
        s_row2.pack(fill=tk.X, pady=2)
        self._reg(s_row2, "card_frame")

        tk.Label(s_row2, text="Infill Pattern:", font=FONT_SMALL).pack(side=tk.LEFT, padx=2)
        self._reg(s_row2.winfo_children()[-1], "label")

        self.slicer_pattern_var = tk.StringVar(value="zigzag")
        pattern_cb = ttk.Combobox(
            s_row2, textvariable=self.slicer_pattern_var, width=28,
            values=["zigzag", "meander", "contour_infill", "crosshatch", "unidirectional"],
            state="readonly"
        )
        pattern_cb.pack(side=tk.LEFT, padx=3)

        self.slicer_reverse_var = tk.BooleanVar(value=True)
        cb_rev = tk.Checkbutton(
            s_row2, text="Bi-Directional Start Reversal (Prevents End Bulge)",
            variable=self.slicer_reverse_var, font=FONT_SMALL
        )
        cb_rev.pack(side=tk.LEFT, padx=6)
        self._reg(cb_rev, "check")

        # Slicer Row 3: Kinematics & Thermal Dwell
        s_row3 = tk.Frame(lf_slicer)
        s_row3.pack(fill=tk.X, pady=2)
        self._reg(s_row3, "card_frame")

        tk.Label(s_row3, text="Weld Feedrate:", font=FONT_SMALL).pack(side=tk.LEFT, padx=2)
        self._reg(s_row3.winfo_children()[-1], "label")
        self.slicer_feedrate_var = tk.StringVar(value="600.0")
        e_sf = tk.Entry(s_row3, textvariable=self.slicer_feedrate_var, width=6, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_sf.pack(side=tk.LEFT, padx=2)
        self._reg(e_sf, "entry")
        tk.Label(s_row3, text="mm/min", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        tk.Label(s_row3, text="Rapid Travel:", font=FONT_SMALL).pack(side=tk.LEFT, padx=4)
        self._reg(s_row3.winfo_children()[-1], "label")
        self.slicer_rapid_var = tk.StringVar(value="3000.0")
        e_sr = tk.Entry(s_row3, textvariable=self.slicer_rapid_var, width=6, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_sr.pack(side=tk.LEFT, padx=2)
        self._reg(e_sr, "entry")
        tk.Label(s_row3, text="mm/min", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        tk.Label(s_row3, text="Inter-Layer Dwell:", font=FONT_SMALL).pack(side=tk.LEFT, padx=4)
        self._reg(s_row3.winfo_children()[-1], "label")
        self.slicer_dwell_var = tk.StringVar(value="10.0")
        e_sd = tk.Entry(s_row3, textvariable=self.slicer_dwell_var, width=5, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_sd.pack(side=tk.LEFT, padx=2)
        self._reg(e_sd, "entry")
        tk.Label(s_row3, text="s", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        tk.Label(s_row3, text="Arc Power:", font=FONT_SMALL).pack(side=tk.LEFT, padx=4)
        self._reg(s_row3.winfo_children()[-1], "label")
        self.slicer_power_var = tk.StringVar(value="3840.0")
        e_spw = tk.Entry(s_row3, textvariable=self.slicer_power_var, width=6, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_spw.pack(side=tk.LEFT, padx=2)
        self._reg(e_spw, "entry")
        tk.Label(s_row3, text="W", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        # Slicer Row 4: Execution Buttons & Output Stats
        s_row4 = tk.Frame(lf_slicer)
        s_row4.pack(fill=tk.X, pady=(4, 2))
        self._reg(s_row4, "card_frame")

        slice_btn = tk.Button(
            s_row4, text="Slice CAD to G-Code Toolpath",
            font=FONT_SMALL_BOLD, command=self._execute_custom_slicer,
            padx=8, pady=2
        )
        slice_btn.pack(side=tk.LEFT, padx=2)
        self._reg(slice_btn, "card_frame")

        self.slicer_summary_label = tk.Label(
            s_row4, text="Slicer Ready. Configure bead parameters and click Slice.",
            font=FONT_MONO, fg="#0d6efd", justify=tk.LEFT, anchor=tk.W
        )
        self.slicer_summary_label.pack(side=tk.LEFT, padx=8)
        self._reg(self.slicer_summary_label, "label")

        # 4. Structured Mesh: Direct Element Size Input (dx, dy, dz)
        lf2 = tk.LabelFrame(scroll_content, text="Define Structured Mesh by Element Size (dx, dy, dz)", font=FONT_SECTION, padx=8, pady=3)
        lf2.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf2, "card_labelframe")
        self._add_info_btn_to_labelframe(lf2, self._show_mesh_info)

        row1 = tk.Frame(lf2)
        row1.pack(fill=tk.X, pady=2)
        self._reg(row1, "card_frame")

        tk.Label(row1, text="Deposit dx:", font=FONT_SMALL).pack(side=tk.LEFT, padx=2)
        self.target_dx_var = tk.StringVar(value="6.67")
        e_dx = tk.Entry(row1, textvariable=self.target_dx_var, width=6, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_dx.pack(side=tk.LEFT, padx=2)
        self._reg(e_dx, "entry")
        tk.Label(row1, text="mm", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        tk.Label(row1, text="dy:", font=FONT_SMALL).pack(side=tk.LEFT, padx=4)
        self.target_dy_var = tk.StringVar(value="1.33")
        e_dy = tk.Entry(row1, textvariable=self.target_dy_var, width=6, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_dy.pack(side=tk.LEFT, padx=2)
        self._reg(e_dy, "entry")
        tk.Label(row1, text="mm", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        tk.Label(row1, text="dz:", font=FONT_SMALL).pack(side=tk.LEFT, padx=4)
        self.target_dz_var = tk.StringVar(value="3.0")
        e_dz = tk.Entry(row1, textvariable=self.target_dz_var, width=6, font=FONT_MONO, relief=tk.SOLID, bd=1)
        e_dz.pack(side=tk.LEFT, padx=2)
        self._reg(e_dz, "entry")
        tk.Label(row1, text="mm", font=FONT_SMALL).pack(side=tk.LEFT, padx=1)

        apply_size_btn = tk.Button(
            row1, text="Apply Size", font=FONT_SMALL_BOLD,
            command=self._apply_element_size_to_divisions
        )
        apply_size_btn.pack(side=tk.LEFT, padx=6)
        self._reg(apply_size_btn, "card_frame")

        # 5. Parametric Divisions (Nx, Ny, Nz)
        lf3 = tk.LabelFrame(scroll_content, text="Mesh Division Counts (Nx, Ny, Nz)", font=FONT_SECTION, padx=8, pady=3)
        lf3.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf3, "card_labelframe")
        self._add_info_btn_to_labelframe(lf3, self._show_mesh_info)

        self._make_param_row(lf3, "Deposit Elements X (Nx_dep):", "nx_dep", 0, "divisions", type_="int")
        self._make_param_row(lf3, "Deposit Elements Y (Ny_dep):", "ny_dep", 1, "divisions", type_="int")
        self._make_param_row(lf3, "Deposit Elements Z (Nz_dep):", "nz_dep", 2, "(= n_layers)", type_="int")
        self._make_param_row(lf3, "Substrate Elements X (Nx_base):", "nx_base", 3, "divisions", type_="int")
        self._make_param_row(lf3, "Substrate Elements Y (Ny_base):", "ny_base", 4, "divisions", type_="int")
        self._make_param_row(lf3, "Substrate Elements Z (Nz_base):", "nz_base", 5, "divisions", type_="int")

        # 6. Live Structured Element Sizing Dashboard
        self.mesh_calc_frame = tk.LabelFrame(scroll_content, text="Live Structured Element Size & Statistics", font=FONT_SECTION, padx=8, pady=4)
        self.mesh_calc_frame.pack(fill=tk.X, padx=8, pady=3)
        self._reg(self.mesh_calc_frame, "card_labelframe")
        self._add_info_btn_to_labelframe(self.mesh_calc_frame, self._show_mesh_info)

        self.mesh_calc_label = tk.Label(
            self.mesh_calc_frame, text="—", font=FONT_MONO, justify=tk.LEFT
        )
        self.mesh_calc_label.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(self.mesh_calc_label, "label")

        preview_btn = tk.Button(
            self.mesh_calc_frame, text="Interactive 3D Mesh Preview (Rotate / Pan / Zoom)",
            font=FONT_SMALL_BOLD, command=self._open_3d_mesh_viewer, padx=8, pady=3
        )
        preview_btn.pack(anchor=tk.W, padx=4, pady=(2, 4))
        self._reg(preview_btn, "card_frame")

        self._update_mesh_ui()

    def _load_sample_cad(self, filepath: str):
        """Quickly load workspace CAD file sample."""
        if os.path.exists(filepath):
            self._handle_cad_stl_selected(filepath, auto_mesh=False)
        else:
            messagebox.showwarning("File Not Found", f"CAD file was not found at:\n{filepath}")

    def _auto_enforce_node_budget(self):
        """Intelligently calculates mesh divisions or meshes loaded CAD file based on geometry rules."""
        try:
            target = int(self.mesh_budget_var.get())
        except Exception:
            target = 2000

        strategy = getattr(self, "mesh_strategy_var", tk.StringVar(value="physics_adaptive")).get()

        # If CAD file mode is active and a CAD file is loaded, mesh the CAD file
        mesh_src = self.mesh_source_var.get()
        cad_path = getattr(self, "loaded_stl_path", "") or self.mesh_file_var.get().strip()
        if mesh_src == "cad_file" and cad_path and os.path.exists(cad_path):
            if cad_path.lower().endswith(".mesh") and hasattr(self, "loaded_stl_path") and self.loaded_stl_path and os.path.exists(self.loaded_stl_path):
                cad_path = self.loaded_stl_path
            self._start_stl_meshing_thread(cad_path)
            return

        try:
            from mesh.intelligent_mesher import calculate_intelligent_structured_mesh
            m = calculate_intelligent_structured_mesh(self._collect_params(), target_nodes=target, strategy=strategy)
            self._param_vars["nx_dep"][0].set(str(m["nx_dep"]))
            self._param_vars["ny_dep"][0].set(str(m["ny_dep"]))
            self._param_vars["nz_dep"][0].set(str(m["nz_dep"]))
            self._param_vars["nx_base"][0].set(str(m["nx_base"]))
            self._param_vars["ny_base"][0].set(str(m["ny_base"]))
            self._param_vars["nz_base"][0].set(str(m["nz_base"]))

            self._update_mesh_calc()
            self._log(
                f"[AUTO-MESH] Applied {strategy.replace('_', ' ').title()} rule:\n"
                f"[AUTO-MESH] Target budget: ~{target} nodes -> Actual: {m['total_nodes']:,} nodes | {m['total_tets']:,} tets\n"
                f"[AUTO-MESH] Deposit Divisions: Nx={m['nx_dep']}, Ny={m['ny_dep']}, Nz={m['nz_dep']} (Aspect Ratio = {m['aspect_ratio_dep']:.2f})\n"
                f"[AUTO-MESH] Substrate Divisions: Nx={m['nx_base']}, Ny={m['ny_base']}, Nz={m['nz_base']} (Aspect Ratio = {m['aspect_ratio_base']:.2f})\n"
            )
            self.status_var.set(f"Status: Auto-Mesh ({m['total_nodes']:,} nodes, AR={m['aspect_ratio_dep']:.2f})")
        except Exception as e:
            messagebox.showerror("Auto-Mesh Error", str(e))

    def _open_3d_mesh_viewer(self):
        """Switch to the dedicated 3D Viewport tab and refresh PyVista embedded renderer."""
        try:
            if hasattr(self, "viewport_tab"):
                self.notebook.select(self.viewport_tab)
                self._refresh_embedded_viewport()
        except Exception as e:
            self._log(f"[3D VIEWER] Error: {e}\n")

    def _open_pyvista_window(self):
        """Launch the standalone hardware-accelerated PyVista GPU 3D window."""
        try:
            from gui.widgets.mesh_viewer_3d import launch_pyvista_gpu_window
            params = self._collect_params()
            launch_pyvista_gpu_window(params)
            self._log("[3D MESH] [OK] Launched native PyVista GPU 3D viewport window.\n")
        except Exception as e:
            self._log(f"[3D MESH] [ERROR] Could not launch PyVista window: {e}\n")
            messagebox.showerror("PyVista GPU Error", str(e))

    def _apply_element_size_to_divisions(self):
        """Calculate Nx, Ny, Nz from target element sizes dx, dy, dz."""
        try:
            # Get geometry dimensions (convert mm if entered as mm)
            def get_dim_m(key):
                val = float(self._param_vars[key][0].get().strip() or "0")
                return val / 1000.0 if val > 1.0 else val

            lx_m = get_dim_m("lx")
            ty_dep_m = get_dim_m("ty_dep")
            hz_base_m = get_dim_m("hz_base")
            ly_base_m = get_dim_m("ly_base")
            n_lay = int(float(self._param_vars["n_layers"][0].get().strip() or "26"))

            dx_mm = float(self.target_dx_var.get().strip() or "6.67")
            dy_mm = float(self.target_dy_var.get().strip() or "1.33")

            nx_dep = max(int(round((lx_m * 1000.0) / dx_mm)), 2)
            ny_dep = max(int(round((ty_dep_m * 1000.0) / dy_mm)), 2)
            nz_dep = n_lay

            nx_base = nx_dep
            ny_base = max(int(round((ly_base_m * 1000.0) / (dy_mm * 3.0))), 4)
            nz_base = max(int(round((hz_base_m * 1000.0) / 2.0)), 2)

            self._param_vars["nx_dep"][0].set(str(nx_dep))
            self._param_vars["ny_dep"][0].set(str(ny_dep))
            self._param_vars["nz_dep"][0].set(str(nz_dep))
            self._param_vars["nx_base"][0].set(str(nx_base))
            self._param_vars["ny_base"][0].set(str(ny_base))
            self._param_vars["nz_base"][0].set(str(nz_base))

            self._update_mesh_calc()
            self._log(f"[MESH] Applied structured sizes dx={dx_mm}mm, dy={dy_mm}mm -> Nx_dep={nx_dep}, Ny_dep={ny_dep}, Nz_dep={nz_dep}\n")
        except Exception as e:
            messagebox.showerror("Mesh Sizing Error", str(e))

    def _update_mesh_calc(self):
        """Update live calculated element sizes displayed in the Mesh tab."""
        if not hasattr(self, "mesh_calc_label"):
            return
        try:
            def get_dim_mm(key):
                if key not in self._param_vars: return 0.0
                val = float(self._param_vars[key][0].get().strip() or "0")
                return val if val > 1.0 else val * 1000.0

            def get_div(key, default=1):
                if key not in self._param_vars: return default
                return max(int(float(self._param_vars[key][0].get().strip() or str(default))), 1)

            lx_mm = get_dim_mm("lx")
            ty_dep_mm = get_dim_mm("ty_dep")
            h_bead_mm = get_dim_mm("h_bead")
            hz_base_mm = get_dim_mm("hz_base")
            ly_base_mm = get_dim_mm("ly_base")

            nx_dep = get_div("nx_dep", 30)
            ny_dep = get_div("ny_dep", 6)
            nz_dep = get_div("nz_dep", 26)

            nx_base = get_div("nx_base", 30)
            ny_base = get_div("ny_base", 20)
            nz_base = get_div("nz_base", 8)

            dx_dep = lx_mm / nx_dep
            dy_dep = ty_dep_mm / ny_dep
            dz_dep = h_bead_mm

            dx_base = lx_mm / nx_base
            dy_base = ly_base_mm / ny_base
            dz_base = hz_base_mm / nz_base

            # Actual substrate plan dimensions and thickness, in mm. The CAD
            # dashboard used to print a hardcoded 150x150x15 plate (and 15 mm
            # while the parameter was 5 mm), which no longer matches the model.
            lx_b_mm = lx_mm
            ly_b_mm = ly_base_mm
            hz_b_mm = hz_base_mm

            ar_dep = max(dx_dep, dy_dep, dz_dep) / max(min(dx_dep, dy_dep, dz_dep), 1e-4)
            ar_base = max(dx_base, dy_base, dz_base) / max(min(dx_base, dy_base, dz_base), 1e-4)

            # Estimated nodes & tets (cube has (nx+1)*(ny+1)*(nz+1) nodes, 6*nx*ny*nz tets)
            nodes_base = (nx_base + 1) * (ny_base + 1) * (nz_base + 1)
            nodes_dep = (nx_dep + 1) * (ny_dep + 1) * (nz_dep + 1)
            tets_base = 6 * nx_base * ny_base * nz_base
            tets_dep = 6 * nx_dep * ny_dep * nz_dep
            total_nodes = nodes_base + nodes_dep
            total_tets = tets_base + tets_dep

            is_cad_active = (
                (hasattr(self, "mesh_source_var") and self.mesh_source_var.get() in ("cad_file", "gcode_file")) or
                bool(getattr(self, "loaded_stl_path", None))
            )
            cad_p = getattr(self, "loaded_stl_path", None) or (self.mesh_file_var.get().strip() if hasattr(self, "mesh_file_var") else None)
            if is_cad_active and cad_p and os.path.exists(cad_p):
                cad_name = os.path.basename(cad_p)
                info = (
                    f"Imported CAD / Toolpath: {cad_name}\n"
                    f"Layer Slicing Count:     {nz_dep} Layers (Layer Height dz = {dz_dep:.2f} mm | Bead Width dy = {dy_dep:.2f} mm)\n"
                    f"Substrate Base Plate:    {lx_b_mm:.1f} x {ly_b_mm:.1f} x {hz_b_mm:.1f} mm (dx = {dx_base:.2f} mm, dy = {dy_base:.2f} mm, dz = {dz_base:.2f} mm)\n"
                    f"Grid Structure:          Gmsh 4.15 OCCT Volume Mesh + Substrate Tetrahedra Split\n"
                    f"Estimated Discretization:~{total_nodes:,} Nodes  |  ~{total_tets:,} Tetrahedra"
                )
            else:
                info = (
                    f"Deposit Bead Element:   dx = {dx_dep:.2f} mm | dy = {dy_dep:.2f} mm | dz = {dz_dep:.2f} mm  (AR = {ar_dep:.2f})\n"
                    f"Substrate Element:      dx = {dx_base:.2f} mm | dy = {dy_base:.2f} mm | dz = {dz_base:.2f} mm  (AR = {ar_base:.2f})\n"
                    f"Grid Structure:         Structured Cartesian Hexahedra -> 6-Tet Split\n"
                    f"Mesh Scale:             ~{total_nodes:,} Nodes  |  ~{total_tets:,} Tetrahedra"
                )
            self.mesh_calc_label.config(text=info)

            # Trigger dynamic 3D viewport refresh if tab is active
            if hasattr(self, "viewport") or hasattr(self, "viewport_3d"):
                try:
                    params_now = self._collect_params()
                    if hasattr(self, "loaded_stl_path") and self.loaded_stl_path and os.path.exists(self.loaded_stl_path):
                        params_now["stl_file"] = self.loaded_stl_path
                    vp = getattr(self, "viewport", getattr(self, "viewport_3d", None))
                    if vp:
                        vp.update_mesh(params_now)
                except Exception:
                    pass
        except Exception:
            pass

    def _update_heat_calc(self):
        """Calculate and display real-time Heat Input, Wire Feed Rate, and Power based on process inputs."""
        if not hasattr(self, "heat_calc_label"):
            return
        try:
            def get_val(key, default):
                if key not in self._param_vars:
                    return float(self.params.get(key, default))
                raw = self._param_vars[key][0].get().strip()
                return float(raw) if raw else float(default)

            I = get_val("I_weld", 180.0)
            U = get_val("U_weld", 20.0)
            eta = get_val("eta_arc", 0.8)
            v_scan_val = get_val("v_scan", 8.0)
            v_scan_mm_s = v_scan_val if v_scan_val > 0.1 else v_scan_val * 1000.0
            v_travel_mm_min = v_scan_mm_s * 60.0

            v_wire_m_min = get_val("v_wire", 5.0)
            d_wire_val = get_val("d_wire", 1.2)
            d_wire_mm = d_wire_val * 1000.0 if d_wire_val < 0.05 else d_wire_val

            w_bead_val = get_val("ty_dep", 8.0)
            w_bead_mm = w_bead_val if w_bead_val > 0.1 else w_bead_val * 1000.0
            h_bead_val = get_val("h_bead", 3.0)
            h_bead_mm = h_bead_val if h_bead_val > 0.1 else h_bead_val * 1000.0

            # 1. Gross & Net Electrical/Thermal Power
            p_gross_w = U * I
            p_gross_kw = p_gross_w / 1000.0
            p_net_w = eta * U * I
            p_net_kw = p_net_w / 1000.0

            # 2. Linear Heat Input per unit length
            if v_scan_mm_s > 0:
                h_net_j_mm = p_net_w / v_scan_mm_s
                h_net_kj_mm = h_net_j_mm / 1000.0
                h_gross_j_mm = p_gross_w / v_scan_mm_s
                h_gross_kj_mm = h_gross_j_mm / 1000.0
            else:
                h_net_j_mm = h_net_kj_mm = h_gross_j_mm = h_gross_kj_mm = 0.0

            # 3. Mass Deposition Rate & Theoretical Bead Area from WFS & Wire Diameter
            # Wire area: A_wire = pi * (d_wire / 2)^2 [mm^2]
            a_wire_mm2 = (math.pi * (d_wire_mm ** 2)) / 4.0
            # Volumetric Wire Feed Rate: Q_wire = A_wire [mm^2] * v_wire [m/min] * 1000 [mm/m] -> mm^3/min
            q_wire_mm3_min = a_wire_mm2 * (v_wire_m_min * 1000.0)

            # Retrieve actual alloy density from Material tab / database
            dep_key = self.material_dep_var.get() if hasattr(self, "material_dep_var") else "SS308L"
            mat_dep = self.materials.get(dep_key, {})
            rho_val = float(mat_dep.get("density", get_val("rho", 7850.0)))

            # Mass deposition rate from wire feed: kg/h and g/s
            m_dep_kg_h = (q_wire_mm3_min * (rho_val * 1e-9)) * 60.0
            m_dep_g_s = (m_dep_kg_h * 1000.0) / 3600.0

            # Theoretical bead area from mass conservation: A_bead = Q_wire / v_travel
            # Q_wire in mm^3/min, v_travel in mm/min -> A_bead in mm^2
            a_bead_theo_mm2 = (q_wire_mm3_min / v_travel_mm_min) if v_travel_mm_min > 0 else 0.0

            # 4. Volumetric Energy Density (J/mm^3) and Specific Melting Enthalpy (kJ/kg)
            bead_area_mm2 = w_bead_mm * h_bead_mm
            e_vol = (h_net_j_mm / bead_area_mm2) if bead_area_mm2 > 0 else 0.0
            e_mass_kj_kg = (p_net_w / (m_dep_g_s / 1000.0) / 1000.0) if m_dep_g_s > 0 else 0.0

            lines = [
                f"Torch Travel Speed (v_travel):    {v_scan_mm_s:8.2f} mm/s   ({v_travel_mm_min:8.1f} mm/min)",
                f"Wire Feed Speed (WFS):            {v_wire_m_min:8.2f} m/min  (Wire ∅ {d_wire_mm:.2f} mm | Aw={a_wire_mm2:.2f} mm²)",
                f"Feedstock Mass Deposition (ṁ):    {m_dep_kg_h:8.2f} kg/h   ({m_dep_g_s:.2f} g/s at {rho_val:.0f} kg/m³ {dep_key})",
                f"Theoretical Bead Area (A_bead):   {a_bead_theo_mm2:8.2f} mm²  (Set CAD Bead Area: {bead_area_mm2:.2f} mm²)",
                f"Net Linear Heat Input (H_net):     {h_net_j_mm:8.2f} J/mm   ({h_net_kj_mm:.3f} kJ/mm | {h_net_j_mm:.1f} kJ/m)",
                f"Gross Linear Heat Input (H_gross): {h_gross_j_mm:8.2f} J/mm   ({h_gross_kj_mm:.3f} kJ/mm)",
                f"Net Thermal Arc Power (P_net):     {p_net_kw:8.3f} kW     ({p_net_w:,.1f} W)",
                f"Gross Electrical Power (P_gross):  {p_gross_kw:8.3f} kW     ({p_gross_w:,.1f} W)",
                f"Specific Melting Energy (E_mass):  {e_mass_kj_kg:8.1f} kJ/kg  (Available arc melting enthalpy per unit mass)",
                f"Volumetric Energy Density (E_vol): {e_vol:8.2f} J/mm³  ({e_vol:.2f} MJ/m³)",
            ]
            self.heat_calc_label.config(text="\n".join(lines))
        except Exception:
            pass

    def _update_mesh_ui(self):
        val = self.mesh_source_var.get()
        if val == "parametric":
            self.file_picker_frame.grid_remove()
        else:
            self.file_picker_frame.grid()

    def _browse_mesh_file(self):
        filetypes = [
            ("3D CAD Models", "*.stl *.step *.stp *.iges *.igs *.ply *.obj"),
            ("STL Files", "*.stl"),
            ("STEP / IGES Files", "*.step *.stp *.iges *.igs"),
            ("Mesh & Toolpath Files", "*.mesh *.msh *.gcode *.nc"),
            ("All Files", "*.*")
        ]
        path = filedialog.askopenfilename(
            filetypes=filetypes,
            title="Select 3D CAD or Mesh File",
            initialdir=str(Path.home() / "Desktop")
        )
        if path:
            self._handle_cad_stl_selected(path, auto_mesh=False)

    def _handle_cad_stl_selected(self, path: str, auto_mesh: bool = False):
        if not path or not os.path.exists(path):
            return

        self.mesh_file_var.set(path)
        self.mesh_source_var.set("cad_file")
        self._update_mesh_ui()

        suffix = Path(path).suffix.lower()
        if suffix in ('.stl', '.ply', '.obj', '.step', '.stp', '.iges', '.igs', '.brep'):
            self.loaded_stl_path = path
            try:
                from gui.widgets.mesh_viewer_3d import load_any_cad_geometry
                verts, faces, info = load_any_cad_geometry(path)
                if verts is None or faces is None or info is None:
                    raise ValueError(f"Could not triangulate {os.path.basename(path)} (empty or unsupported geometry)")
                bb = info["bbox_mm"]
                lx_mm = round(float(bb["x"]), 2)
                ly_mm = round(float(bb["y"]), 2)
                hz_mm = round(float(bb["z"]), 2)

                # Substrate plate: the framework default is 5 mm thick, matching
                # the solver's hz_base and the volume mesher's plate.
                if "lx" in self._param_vars:
                    self._param_vars["lx"][0].set("150.0")
                if "ly_base" in self._param_vars:
                    self._param_vars["ly_base"][0].set("150.0")
                if "hz_base" in self._param_vars:
                    self._param_vars["hz_base"][0].set("5.0")
                # Enforce exactly 3mm layer height and 4mm bead width for imported CAD
                from tkinter import simpledialog
                default_layers = max(int(math.ceil(hz_mm / 3.0)), 1)
                
                # Ask user for layers
                user_layers = simpledialog.askinteger(
                    "CAD Slicing", 
                    f"Geometry loaded: Lx={lx_mm:.1f}mm, Ly={ly_mm:.1f}mm, Lz={hz_mm:.1f}mm\n\nHow many layers do you want to slice this CAD into?\n(Default enforces ~3.0mm height)", 
                    initialvalue=default_layers, 
                    minvalue=1,
                    parent=self.root
                )
                
                cad_layers = user_layers if user_layers is not None else default_layers
                h_bead = 3.0 # Enforce exactly 3.0mm as requested
                
                if "ty_dep" in self._param_vars:
                    self._param_vars["ty_dep"][0].set("4.0") # Enforce exactly 4.0mm width
                    
                if hasattr(self, "slicer_h_var"):
                    self.slicer_h_var.set(f"{h_bead:.3f}")
                if "h_bead" in self._param_vars:
                    self._param_vars["h_bead"][0].set(f"{h_bead:.3f}")
                
                # When slicing CAD, we don't use remelt overlap for layer height calculation
                if "remelt_ratio" in self._param_vars:
                    self._param_vars["remelt_ratio"][0].set("1.0")

                if "n_layers" in self._param_vars:
                    self._param_vars["n_layers"][0].set(str(cad_layers))
                if "nz_dep" in self._param_vars:
                    self._param_vars["nz_dep"][0].set(str(cad_layers))

                scale_applied = info.get("scale_applied", 1.0)
                vol_pct = info.get("vol_ratio_pct", 0.0)
                v_occ = info.get("volume_occupied_mm3", 0.0)
                orig_bb = info.get("orig_bbox_mm", {})
                scale_str = f" | Auto-scaled {scale_applied:.2f}x ({vol_pct:.1f}% enclosure vol)" if scale_applied < 0.999 else f" ({vol_pct:.1f}% enclosure vol)"

                if hasattr(self, "cad_info_label"):
                    self.cad_info_label.config(
                        text=f"Loaded CAD: {os.path.basename(path)} ({info['format']}) | "
                             f"Lx={lx_mm:.1f} mm, Ly={ly_mm:.1f} mm, Lz={hz_mm:.1f} mm{scale_str} | "
                             f"Substrate 150x150x5 mm | {len(verts):,} Verts"
                    )

                self._update_mesh_calc()
                self._update_heat_calc()

                lx_b_mm = float(self._param_vars.get("lx", ["150.0"])[0].get() if "lx" in self._param_vars else 150.0)
                ly_b_mm = float(self._param_vars.get("ly_base", ["150.0"])[0].get() if "ly_base" in self._param_vars else 150.0)
                hz_b_mm = float(self._param_vars.get("hz_base", ["5.0"])[0].get() if "hz_base" in self._param_vars else 5.0)

                orig_str = f" | Original Size: {orig_bb.get('x', lx_mm):.1f} x {orig_bb.get('y', ly_mm):.1f} x {orig_bb.get('z', hz_mm):.1f} mm" if scale_applied < 0.999 else ""
                self._log(
                    f"\n[CAD-ENCLOSURE] Substrate Plate: {lx_b_mm:.1f} x {ly_b_mm:.1f} x {hz_b_mm:.1f} mm\n"
                    f"[CAD-ENCLOSURE] Build Enclosure Volume: 150.0 x 150.0 x 150.0 mm (3.375 L)\n"
                    f"[CAD-ENCLOSURE] Max Allowable Occupied Volume (75% Limit): 2.531 L (2,531,250 mm³)\n"
                    f"[CAD-IMPORT] Loaded CAD Model: {os.path.basename(path)} ({info['format']}){orig_str}\n"
                    f"[CAD-IMPORT] Full Path: {path}\n"
                    f"[CAD-IMPORT] Scale Applied: {scale_applied:.4f}x -> Occupied Volume: {v_occ:,.1f} mm³ ({vol_pct:.1f}% of enclosure)\n"
                    f"[CAD-IMPORT] Placed Bounding Box: Lx = {lx_mm:.2f} mm | Ly = {ly_mm:.2f} mm | Lz = {hz_mm:.2f} mm\n"
                    f"[CAD-IMPORT] Base Plate Placement: Centered at X=75.0, Y=75.0 mm | Resting on Substrate Top at Z={hz_b_mm:.1f} mm\n"
                )
                self.status_var.set(f"Status: CAD Placed ({lx_mm:.0f}x{ly_mm:.0f}x{hz_mm:.0f} mm | {vol_pct:.1f}% enclosure vol)")
            except Exception as e:
                self._log(f"[CAD-IMPORT] [WARNING] Could not parse CAD geometry: {e}\n")

        # Live update 3D viewport tab immediately
        params_now = self._collect_params()
        params_now["stl_file"] = path
        params_now["mesh_file"] = path
        vp = getattr(self, "viewport", getattr(self, "viewport_3d", None))
        if vp:
            vp.update_mesh(params_now)

        if auto_mesh:
            self._start_stl_meshing_thread(path)

    def _start_stl_meshing_thread(self, cad_path: str):
        target_nodes = 2000
        try:
            target_nodes = int(self.mesh_budget_var.get())
        except Exception:
            pass

        # The meshed substrate MUST be the plate the solver models: the
        # generated .edp marks everything at z <= HzBase as substrate. Passing
        # the UI's plate/lx/ly/layer height through keeps the two in step (the
        # mesher's own 5 mm / 150 mm defaults match the framework default plate).
        def _dim_mm(key, default_mm):
            try:
                val = float(self._param_vars[key][0].get().strip() or default_mm)
            except Exception:
                return float(default_mm)
            # UI entries are millimetres; anything between 0 and 1 was typed in
            # SI metres and is scaled up to match.
            return val * 1000.0 if 0.0 < val < 1.0 else val

        sub_hz_mm = _dim_mm("hz_base", 5.0)
        sub_lx_mm = _dim_mm("lx", 150.0)
        sub_ly_mm = _dim_mm("ly_base", 150.0)
        layer_height_mm = _dim_mm("h_bead", 3.0)

        self._log(f"[CAD-IMPORT] Target volume mesh size: ~{target_nodes} nodes (Gmsh 4.15 + OCCT volume pipeline)\n")
        self._log(f"[CAD-IMPORT] Substrate plate for the mesh: {sub_lx_mm:.1f} x {sub_ly_mm:.1f} x {sub_hz_mm:.1f} mm | layer height {layer_height_mm:.2f} mm\n")
        self.status_var.set("Status: Meshing CAD...")

        def run_mesh():
            try:
                from mesh.intelligent_mesher import import_and_mesh_cad_intelligent
                outdir = "./waam_output"
                os.makedirs(outdir, exist_ok=True)
                out_mesh = os.path.join(outdir, f"imported_{Path(cad_path).stem}.mesh")
                result = import_and_mesh_cad_intelligent(
                    cad_path, out_mesh, target_nodes=target_nodes, refine_thin_features=True,
                    layer_height=layer_height_mm, sub_hz_mm=sub_hz_mm,
                    sub_lx_mm=sub_lx_mm, sub_ly_mm=sub_ly_mm)

                for w in result.get("warnings", []):
                    self.root.after(0, self._log, f"[CAD] {w}\n")

                bb = result["bbox_mm"]
                self.root.after(0, self._log,
                    f"[CAD-IMPORT] [OK] Mesh complete: {result['n_nodes']:,} nodes  |  {result['n_tets']:,} tets\n"
                    f"[CAD-IMPORT]     Format: {result.get('format', 'CAD')}\n"
                    f"[CAD-IMPORT]     Bounding box: X={bb['x']:.1f}  Y={bb['y']:.1f}  Z={bb['z']:.1f} mm\n"
                    f"[CAD-IMPORT]     Mesh file: {result['mesh_path']}\n"
                )

                # Auto-populate mesh file path in UI (stays in cad_file mode: mesh came from CAD)
                self.root.after(0, lambda: self.mesh_file_var.set(result["mesh_path"]))
                self.root.after(0, lambda: self.mesh_source_var.set("cad_file"))
                self.root.after(0, lambda: self.status_var.set("Status: CAD Meshed"))

                # Live update 3D viewport tab with imported CAD geometry and mesh
                params_cad = self._collect_params()
                params_cad["mesh_file"] = result["mesh_path"]
                params_cad["stl_file"] = cad_path
                vp = getattr(self, "viewport", getattr(self, "viewport_3d", None))
                if vp:
                    self.root.after(0, lambda: vp.update_mesh(params_cad))

            except Exception as e:
                self.root.after(0, self._log, f"[CAD-IMPORT] [ERROR] Meshing error: {e}\n")
                self.root.after(0, lambda: self.status_var.set("Status: Meshing Failed"))

        import threading
        threading.Thread(target=run_mesh, daemon=True).start()

    def _execute_custom_slicer(self):
        """Execute customized toolpath generation based on user G-code parameters."""
        # Prefer the original imported 3D CAD/STL model for slicing
        loaded_cad = getattr(self, "loaded_stl_path", "")
        cad_path = ""
        if loaded_cad and os.path.exists(loaded_cad) and Path(loaded_cad).suffix.lower() in ('.stl', '.step', '.stp', '.iges', '.igs', '.ply', '.obj'):
            cad_path = loaded_cad
        else:
            mf = self.mesh_file_var.get().strip()
            if mf and os.path.exists(mf):
                cad_path = mf
        if not cad_path or not os.path.exists(cad_path):
            cad_path = getattr(self, "loaded_stl_path", "")
        if not cad_path or not os.path.exists(cad_path):
            cad_path = filedialog.askopenfilename(
                filetypes=[("3D CAD / STL / Mesh", "*.stl *.step *.stp *.iges *.igs *.ply *.obj *.mesh"), ("All", "*.*")],
                title="Select CAD Model to Slice",
                initialdir=str(Path.home() / "Desktop")
            )
            if not cad_path:
                return


        try:
            h_bead = float(self.slicer_h_var.get())
            w_bead = float(self.slicer_w_var.get())
            # Slicer places the part on the substrate top, so it MUST get the
            # same thickness the solver meshes. Hardcoding 15 mm here (while the
            # builder used 5 mm) floated every sliced part 10 mm above the plate.
            hz_base_mm = (self._f(self._param_vars["hz_base"][0], 5.0)
                          if "hz_base" in self._param_vars else 5.0)
            infill = self.slicer_pattern_var.get()
            overlap = float(self.slicer_overlap_var.get())
            perims = int(self.slicer_perim_var.get())
            feedrate = float(self.slicer_feedrate_var.get())
            rapid = float(self.slicer_rapid_var.get())
            dwell = float(self.slicer_dwell_var.get())
            power = float(self.slicer_power_var.get())
            start_rev = bool(self.slicer_reverse_var.get())

            # Place generated G-code and CAD files directly inside active timestamped run directory
            run_dir = self._get_or_create_run_dir(is_new_run=False)
            os.makedirs(run_dir, exist_ok=True)

            cad_stem = Path(cad_path).stem
            out_gcode = os.path.join(run_dir, f"{cad_stem}.gcode")

            # Also copy CAD file into run_dir for complete self-contained reproducibility
            cad_dest = os.path.join(run_dir, os.path.basename(cad_path))
            if os.path.abspath(cad_path) != os.path.abspath(cad_dest):
                try:
                    import shutil
                    shutil.copy2(cad_path, cad_dest)
                except Exception:
                    pass

            self._log(
                f"\n[SLICER] Slicing CAD: {os.path.basename(cad_path)}\n"
                f"[SLICER] Target G-code in run folder: {out_gcode}\n"
                f"[SLICER] Layer height: {h_bead} mm | Bead width: {w_bead} mm | Infill: {infill}\n"
                f"[SLICER] Overlap: {overlap}% | Perimeters: {perims} | Feedrate: {feedrate} mm/min\n"
                f"[SLICER] Dwell cooling: {dwell} s | Power: {power} W | Reverse starts: {start_rev}\n"
            )

            from mesh.slicer_api import slice_geometry_to_toolpath
            tp = slice_geometry_to_toolpath(
                geometry_file=cad_path,
                output_gcode=out_gcode,
                layer_height=h_bead,
                bead_width=w_bead,
                sub_hz=hz_base_mm,
                infill_pattern=infill,
                overlap_pct=overlap,
                perimeters=perims,
                feedrate=feedrate,
                rapid_feedrate=rapid,
                power=power,
                dwell_time=dwell,
                start_reverse=start_rev
            )

            # Save toolpath summary metadata in run_dir
            summary_path = os.path.join(run_dir, "toolpath_summary.json")
            try:
                with open(summary_path, "w") as sf:
                    json.dump(tp, sf, indent=2, default=str)
            except Exception:
                pass

            n_sliced = int(tp.get('n_layers', 1))
            if "n_layers" in self._param_vars:
                self._param_vars["n_layers"][0].set(str(n_sliced))
            if "nz_dep" in self._param_vars:
                self._param_vars["nz_dep"][0].set(str(n_sliced))
            if hasattr(self, "target_dz_var"):
                self.target_dz_var.set(str(h_bead))

            self.mesh_file_var.set(out_gcode)
            self.mesh_source_var.set("gcode_file")
            self._update_mesh_ui()
            self._update_mesh_calc()

            summary = (
                f"Generated {tp['n_layers']} layers toolpath ({tp['total_length_m']:.2f} m)\n"
                f"Estimated Build Time: {tp['estimated_time_min']:.1f} min | Deposited Mass: {tp['bead_mass_kg']:.3f} kg\n"
                f"Arc Energy: {tp['energy_kj']:.1f} kJ | G-code: {os.path.basename(out_gcode)}"
            )
            if hasattr(self, "slicer_summary_label"):
                self.slicer_summary_label.config(text=summary)

            self._log(f"[SLICER] [OK] Success: {summary}\n")
            self.status_var.set(f"Status: Sliced ({tp['n_layers']} layers, {tp['total_length_m']:.1f}m)")

            # Auto load toolpath in viewport
            params_tp = self._collect_params()
            params_tp["gcode_file"] = out_gcode
            params_tp["stl_file"] = cad_path
            params_tp["n_layers"] = n_sliced
            vp = getattr(self, "viewport", getattr(self, "viewport_3d", None))
            if vp:
                vp.update_mesh(params_tp)

            messagebox.showinfo(
                "Toolpath Generation Complete",
                f"Successfully generated custom WAAM toolpath:\n\n"
                f"File: {out_gcode}\n"
                f"Layers: {tp['n_layers']}\n"
                f"Total Length: {tp['total_length_m']} m\n"
                f"Estimated Time: {tp['estimated_time_min']} min\n"
                f"Deposited Mass: {tp['bead_mass_kg']} kg\n"
                f"Arc Energy: {tp['energy_kj']} kJ"
            )
        except Exception as e:
            self._log(f"[SLICER] [ERROR] Slicing failed: {e}\n")
            messagebox.showerror("Slicer Error", str(e))

    def _run_slicer_dialog(self):
        """Legacy slicer dialog forwarding to custom slicer."""
        self._execute_custom_slicer()

    def _build_physics_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Physics")
        self._reg(tab, "card_frame")

        lf1 = tk.LabelFrame(tab, text="Select Coupled Physics Modules", font=FONT_SECTION, padx=8, pady=4)
        lf1.pack(fill=tk.X, padx=8, pady=4)
        self._reg(lf1, "card_labelframe")
        self._add_info_btn_to_labelframe(lf1, self._show_physics_info)

        self.rad_var = tk.BooleanVar(value=True)
        cb1 = tk.Checkbutton(lf1, text="Surface Radiation (Stefan-Boltzmann: ε·σ·(T⁴ - T_amb⁴)) [W/m²]", variable=self.rad_var, font=FONT_SMALL)
        cb1.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb1, "check")

        self.latent_var = tk.BooleanVar(value=True)
        cb2 = tk.Checkbutton(lf1, text="Latent Heat of Fusion (Mushy zone apparent C_p) [J/(kg·K)]", variable=self.latent_var, font=FONT_SMALL)
        cb2.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb2, "check")

        self.mech_var = tk.BooleanVar(value=True)
        cb3 = tk.Checkbutton(lf1, text="Residual Stress & Substrate Distortion (J₂ Plasticity / Inherent Strain) [MPa]", variable=self.mech_var, font=FONT_SMALL)
        cb3.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb3, "check")

        self.micro_var = tk.BooleanVar(value=True)
        cb4 = tk.Checkbutton(lf1, text="Microstructure Evolution (G/R ratio + JMAK kinetics)", variable=self.micro_var, font=FONT_SMALL)
        cb4.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb4, "check")

        self.p48_buoyancy_var = tk.BooleanVar(value=True)
        cb5 = tk.Checkbutton(lf1, text="Buoyancy Body Force (P48 / Boussinesq: F_b = -ρ·g·β_t·(T - T_l)) [N/m³]",
                             variable=self.p48_buoyancy_var, font=FONT_SMALL)
        cb5.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb5, "check")

        self.p48_droplet_var = tk.BooleanVar(value=True)
        cb6 = tk.Checkbutton(lf1, text="Droplet Impingement Enthalpy & Marangoni Shear (P48: Q_drop + ∂γ/∂T · ∇_s T) [W / N/m²]",
                             variable=self.p48_droplet_var, font=FONT_SMALL)
        cb6.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb6, "check")

        self.foam_var = tk.BooleanVar(value=False)
        cb7 = tk.Checkbutton(lf1, text="Coupling to OpenFOAM 2412 (3D Multiphase Free-Surface CFD)", variable=self.foam_var, font=FONT_SMALL)
        cb7.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb7, "check")

        self.enable_ultrasonic_var = tk.BooleanVar(value=False)
        cb8 = tk.Checkbutton(
            lf1,
            text="Ultrasonic Vibration Assistance & Acoustic Softening (UVA / 20 kHz / Blaha Effect)",
            variable=self.enable_ultrasonic_var,
            font=FONT_SMALL,
            command=self._on_ultrasonic_physics_toggle
        )
        cb8.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb8, "check")

        # Mechanical Residual Distortion & Inherent Strain Card
        lf_mech = tk.LabelFrame(tab, text="Mechanical Distortion & Boundary Scheme", font=FONT_SECTION, padx=8, pady=4)
        lf_mech.pack(fill=tk.X, padx=8, pady=4)
        self._reg(lf_mech, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_mech, self._show_mechanical_info)

        # Boundary condition dropdown
        bc_lbl = tk.Label(lf_mech, text="Boundary Condition Scheme:", font=FONT_LABEL)
        bc_lbl.grid(row=0, column=0, sticky=tk.W, padx=(8, 4), pady=2)
        self._reg(bc_lbl, "label")

        bc_options = [
            "Isostatic 3-2-1 / Elastic Support (Recommended)",
            "Clamped Fixture (Full Baseplate Fixed)",
            "Cantilever (x=0 Face Fixed)"
        ]
        self.mech_bc_var = tk.StringVar(value=bc_options[0])
        bc_combo = ttk.Combobox(lf_mech, textvariable=self.mech_bc_var, values=bc_options, state="readonly", width=42)
        bc_combo.grid(row=0, column=1, columnspan=2, padx=4, pady=2, sticky=tk.W)

        self._make_param_row(lf_mech, "Fixture Stiffness Ratio (vertical):", "fixture_stiffness_ratio_z", 4, "× k_bend")
        self._make_param_row(lf_mech, "Fixture Stiffness Ratio (lateral):", "fixture_stiffness_ratio_xy", 5, "× k_bend")

        # Incremental mechanics: how often the structural solve runs
        lf_mech2 = tk.LabelFrame(tab, text="Structural Solve Cadence & Equilibrium",
                                 font=FONT_SECTION, padx=8, pady=4)
        lf_mech2.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_mech2, "card_labelframe")

        cad_row_frame = tk.Frame(lf_mech2)
        cad_row_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=4, pady=2)
        self._reg(cad_row_frame, "card_frame")

        lbl_cad = tk.Label(cad_row_frame, text="Solve Cadence:", font=FONT_LABEL)
        lbl_cad.pack(side=tk.LEFT, padx=(0, 4))
        self._reg(lbl_cad, "label")

        self.mech_cadence_var = tk.StringVar(value="interpass")
        cadence_combo = ttk.Combobox(
            cad_row_frame, textvariable=self.mech_cadence_var, state="readonly", width=18,
            values=["interpass", "layer_n", "weld"])
        cadence_combo.pack(side=tk.LEFT, padx=4)

        info_btn = tk.Button(
            cad_row_frame, text="ⓘ Info", font=FONT_SMALL_BOLD,
            command=self._show_solve_cadence_info, padx=6, pady=1,
            relief=tk.FLAT, bd=0, highlightthickness=0
        )
        info_btn.pack(side=tk.LEFT, padx=6)
        self._reg(info_btn, "button")

        lbl_hint = tk.Label(
            lf_mech2,
            text="inter-pass = once per layer (fast) · layer_n = every N steps · weld = every step (Abaqus-like)",
            font=FONT_SMALL, fg="#64748b"
        )
        lbl_hint.grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=4, pady=(1, 3))
        self._reg(lbl_hint, "label_dim")

        self._make_param_row(lf_mech2, "Weld Steps Between Solves (N):", "mech_solve_every_n_weld", 2, "steps", type_="int")
        self._make_param_row(lf_mech2, "Max Equilibrium Iterations:", "mech_max_newton_iter", 3, "iters", type_="int")
        self._make_param_row(lf_mech2, "Displacement Convergence Tol:", "mech_newton_tol", 4, "[-]")
        self._make_param_row(lf_mech2, "Force Residual Tolerance:", "mech_newton_tol_force", 5, "[-]")
        self._make_param_row(lf_mech2, "Isotropic Hardening H':", "h_prime_iso", 6, "Pa")

        self.springback_var = tk.BooleanVar(value=True)
        cb_spring = ttk.Checkbutton(lf_mech2, text="Unclamp & solve springback after cooling",
                                    variable=self.springback_var)
        cb_spring.grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=4, pady=2)

    def _add_info_btn_to_labelframe(self, lf, command):
        """Place an Info button in the top right corner of a LabelFrame."""
        btn = tk.Button(
            lf, text="ⓘ Info", font=FONT_SMALL_BOLD,
            command=command, padx=6, pady=1,
            relief=tk.FLAT, bd=0, highlightthickness=0
        )
        btn.place(relx=1.0, rely=0.0, anchor="ne", x=-6, y=2)
        self._reg(btn, "button")
        return btn

    def _show_info_dialog(self, title: str, heading: str, text_blocks: list):
        """Generic info dialog generator with rich typography and scrollable text."""
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.geometry("760x640")
        dlg.minsize(680, 500)
        dlg.transient(self.root)
        dlg.grab_set()

        canvas = tk.Canvas(dlg, highlightthickness=0, bg=self.colors["bg_card"])
        scroll = ttk.Scrollbar(dlg, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas, bg=self.colors["bg_card"], padx=18, pady=14)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        cw = canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        title_lbl = tk.Label(
            frame, text=heading,
            font=FONT_TITLE, fg=self.colors["accent"], bg=self.colors["bg_card"]
        )
        title_lbl.pack(anchor=tk.W, pady=(0, 10))

        for sect_title, body_text in text_blocks:
            card = tk.LabelFrame(
                frame, text=f"  {sect_title}  " if sect_title else "",
                font=FONT_SECTION, fg="#2563eb", bg=self.colors["bg_card"],
                padx=12, pady=8
            )
            card.pack(fill=tk.X, pady=6)
            b = tk.Label(
                card, text=body_text,
                font=FONT_LABEL, fg=self.colors["fg"], bg=self.colors["bg_card"],
                justify=tk.LEFT, wraplength=690
            )
            b.pack(anchor=tk.W, fill=tk.X)

        btn_close = tk.Button(
            frame, text="Close Guide", font=FONT_BTN,
            command=dlg.destroy, padx=14, pady=5, relief=tk.FLAT,
            bg=self.colors["button"], fg=self.colors["button_fg"]
        )
        btn_close.pack(anchor=tk.E, pady=(12, 4))

    def _show_substrate_info(self):
        self._show_info_dialog(
            "Substrate (Base Plate) Engineering Guide",
            "Substrate Base Plate Engineering & Boundary Mechanics Guide",
            [
                ("1. Coordinate Reference Frame & Dimensions (Lx, Ly_base, hz_base)",
                 "• Global Origin (0, 0, 0): Located at the bottom-corner of the substrate base plate.\n"
                 "• Longitudinal Axis (X): Runs along the length Lx (mm). Deposition toolpath travels along X.\n"
                 "• Transverse Axis (Y): Spans the base width Ly_base (mm). Centerline is located at Yc = Ly_base / 2.\n"
                 "• Vertical Elevation (Z): Plate thickness hz_base (mm). The substrate top surface and fusion boundary sit at Z = hz_base."),
                ("2. Thermal Conductive Heat Sinking & Biot Number",
                 "• Governing Heat Equation: ρ·Cp·(∂T/∂t) = ∇·(k∇T). The substrate acts as the primary heat sink.\n"
                 "• Biot Number (Bi = h·Lc/k): For thick plates (Bi << 0.1 internally), 3D conduction dominates over surface convection.\n"
                 "• Cooling Gradient: Thicker substrates provide rapid 3D conductive heat extraction away from the weld bead, resulting in steeper cooling rates (higher dT/dt) and narrower heat-affected zones (HAZ)."),
                ("3. Mechanical Bending Rigidity & Warpage Resistance",
                 "• Flexural Rigidity D = E·hz_base³ / [12·(1 - ν²)]. Plate bending resistance scales cubically with thickness.\n"
                 "• Thin Baseplates (< 8 mm): Undergo significant longitudinal bowing and transverse angular distortion due to weld shrinkage moments.\n"
                 "• Thick Baseplates (≥ 12 mm): Strongly resist global out-of-plane distortion but accumulate higher tensile residual stress (up to yield strength σy) across the deposit/substrate fusion interface."),
                ("4. Preheating & Boundary Fixturing Recommendations",
                 "• Substrate Preheating: Raising initial substrate temperature (e.g. 100 - 150 °C for steel) decreases cooling rates, lowers peak thermal stress, and mitigates cold cracking.\n"
                 "• Clamping Interaction: Fixture springs restrain the plate underside during deposition; releasing clamps simulates post-deposition elastic springback.")
            ]
        )

    def _show_fusion_info(self):
        self._show_info_dialog(
            "WAAM Deposit Geometry & Fusion Model Guide",
            "Deposit Bead Morphology, Layer Slicing & Fusion Retention Guide",
            [
                ("1. Bead Width (w_bead) & Initial Layer Height (h1)",
                 "• Single Bead Width (ty_dep): Transverse width of the deposited track (typically 4.0 - 10.0 mm for GMAW/WAAM).\n"
                 "• Layer 1 Height (h_bead): Vertical height of the initial layer deposited directly on the bare substrate plate.\n"
                 "• Aspect Ratio (w_bead / h1): Recommended between 1.5 and 3.0. Ratios < 1.5 risk bead humping or collapse; ratios > 3.5 cause excessive lateral spreading."),
                ("2. Remelt / Fusion Retention Physics",
                 "• Remelt Ratio (β_remelt): Fraction of preceding layer re-melted by the incoming electric arc (typical range: 0.70 - 0.85, default 0.80).\n"
                 "• Subsequent Layer Height Increment: Δh_layer = h_bead · (1.0 - β_remelt). For h_bead = 4.0 mm and β = 0.80, Δh = 0.80 mm (or nominal bead build ~ 1.32 mm with weld overlap).\n"
                 "• Physics Impact: High remelt ratios enhance interlayer bonding and reduce lack-of-fusion voids, but reduce build rate and increase heat accumulation."),
                ("3. Bi-Directional Dimensioning Drivers (N_layer vs H_build)",
                 "• Strict Bi-Directional Coupling: Either Total Layer Count (N_layer) or Target Wall Height (H_build) can drive the build.\n"
                 "• Total Build Height Formula: H_build = h1 + (N_layer - 1) · Δh_layer.\n"
                 "• Automatic Synchronization: Typing in either field auto-updates the opposite field without locking entries."),
                ("4. Deposition Scanning Patterns",
                 "• Unidirectional: Torch deposits from X=0 to X=Lx on every pass. Leads to uneven height accumulation and end-stop crater build-up.\n"
                 "• Bidirectional (Reverse): Torch reverses direction on alternate layers (Layer 1: 0→Lx, Layer 2: Lx→0). Ensures symmetric temperature distribution, uniform wall height, and balanced residual stress.")
            ]
        )

    def _show_process_info(self):
        self._show_info_dialog(
            "Arc & Energy Process Parameters Guide",
            "Electric Arc Power, Heat Input & Feedstock Kinematics Guide",
            [
                ("1. Electrical Arc Power (P_gross = U · I)",
                 "• Welding Current (I_weld): Dictates arc force, penetration depth, and wire burn-off rate (typical: 120 - 280 A).\n"
                 "• Welding Voltage (U_weld): Dictates arc length and bead spreading width (typical: 18 - 28 V).\n"
                 "• Gross Arc Power: P_gross = U · I [Watts]. For 200 A and 24 V, P_gross = 4800 W."),
                ("2. Arc Thermal Transfer Efficiency (η_arc)",
                 "• Net Workpiece Power: P_net = η_arc · U · I [Watts].\n"
                 "• GMAW / MIG: η_arc ≈ 0.75 - 0.85 (default 0.80).\n"
                 "• GTAW / TIG: η_arc ≈ 0.60 - 0.70.\n"
                 "• Plasma Arc: η_arc ≈ 0.65 - 0.75.\n"
                 "• Unaccounted energy is lost via arc radiation and droplet spatter."),
                ("3. Net Linear Heat Input (H_net)",
                 "• Formula: H_net = (η_arc · U · I) / v_scan [J/mm or kJ/mm].\n"
                 "• For P_net = 3840 W and v_scan = 10 mm/s: H_net = 384 J/mm (0.384 kJ/mm).\n"
                 "• Metallurgical Significance: Higher linear heat input slows cooling rates (Δt_8/5), coarsens austenite grain size, increases dendrite arm spacing, and elevates angular distortion."),
                ("4. Wire Feedstock Kinematics & Deposition Rate",
                 "• Wire Feed Speed (v_wire): Governs volumetric mass deposition rate (typical: 3.0 - 8.0 m/min).\n"
                 "• Wire Diameter (d_wire): Standard feedstock wire diameter (typical: 1.0 or 1.2 mm).\n"
                 "• Mass Deposition Rate: ṁ = ρ_wire · (π·d_wire² / 4) · v_wire · 3600 [kg/h].")
            ]
        )

    def _show_heat_calc_info(self):
        self._show_info_dialog(
            "Heat Input & Mass Deposition Audit Guide",
            "Real-Time Thermal Audit & Deposition Physics Guide",
            [
                ("1. Real-Time Linear Heat Input (H_in)",
                 "• Formula: H_in = (η · U · I) / v_scan [J/mm].\n"
                 "• Live Readout: Continuously recomputed whenever current, voltage, travel speed, or efficiency changes.\n"
                 "• Standard WAAM Range: 250 - 650 J/mm for steel and nickel alloys; 150 - 350 J/mm for aluminium alloys."),
                ("2. Volumetric Energy Density (E_vol)",
                 "• Formula: E_vol = P_net / (A_bead · v_scan) = P_net / V_dot [J/mm³].\n"
                 "• Quantifies energy concentration delivered per unit volume of deposited metal.\n"
                 "• Excessive E_vol (> 80 J/mm³) causes melt pool sagging, excessive penetration, and loss of bead profile."),
                ("3. Mass Deposition & Theoretical Bead Cross-Section",
                 "• Theoretical Bead Area: A_theo = (π · d_wire² · v_wire) / (4 · v_scan) [mm²].\n"
                 "• Mass Deposition Rate: ṁ = ρ · A_theo · v_scan [kg/h].\n"
                 "• Compares programmed wire feed rate against the geometric bead cross-section (w_bead · h_layer)."),
                ("4. Finite Element Energy Conservation Audit",
                 "• The FreeFEM++ solver integrates the moving volumetric source across 3D element quadrature: ∫∫∫ q(x,y,z,t) dΩ.\n"
                 "• Validates that injected thermal energy strictly equals the programmed arc power P_net within < 0.1% numerical tolerance.")
            ]
        )

    def _show_heatsource_info(self):
        self._show_info_dialog(
            "Analytical Heat Source Models Guide",
            "Volumetric & Surface Heat Source Formulations Guide",
            [
                ("1. Goldak Double-Ellipsoid Heat Source (Standard Arc/WAAM)",
                 "• Formulation: Combines two semi-ellipsoids with shared semi-axes b (half-width) and c (depth):\n"
                 "  Front (x > 0): q_f(x,y,z) = [6√3·f_f·Q / (π√π·af·b·c)] · exp(-3x²/af² - 3y²/b² - 3z²/c²)\n"
                 "  Rear (x < 0):  q_r(x,y,z) = [6√3·f_r·Q / (π√π·ar·b·c)] · exp(-3x²/ar² - 3y²/b² - 3z²/c²)\n"
                 "• Power Distribution: f_f + f_r = 2.0 (typical: f_f = 0.60, f_r = 1.40).\n"
                 "• Recommended Dimensions: af ≈ 0.8 · w_bead, ar ≈ 2.0 · w_bead, b ≈ 0.5 · w_bead, c ≈ h_bead."),
                ("2. P48 Dual-Component Arc + Droplet Impingement",
                 "• Component 1 (Surface Gaussian Arc Flux): Delivers ~60% of power to the top surface via electron/plasma conduction.\n"
                 "• Component 2 (Volumetric Droplet Transfer): Transports ~40% of power via superheated liquid metal droplets directly into the weld pool root.\n"
                 "• Physical Fidelity: Accurately reproduces deep finger penetration characteristic of GMAW spray transfer."),
                ("3. Gaussian Volumetric & Surface Models",
                 "• 3D Cylindrical Gaussian: q(r,z) = [3Q / (π·r0²·H)] · exp(-3r²/r0²) with uniform or linear depth decay.\n"
                 "• 2D Surface Gaussian: q(r) = [3Q / (π·r0²)] · exp(-3r²/r0²), optimal for laser beam deposition or thin foils."),
                ("4. Conical Frustum & Uniform Band",
                 "• Conical Frustum: Linearly tapering Gaussian radius r(z) from top surface to root, suitable for plasma keyhole welding.\n"
                 "• Uniform Moving Band: Homogenized rectangular heat flux for wide weaving passes or high-speed oscillating torches.")
            ]
        )

    def _show_material_info(self):
        self._show_info_dialog(
            "Alloy Metallurgy & Multi-Material Guide",
            "Constitutional Metallurgy & Dissimilar Alloy Transitions Guide",
            [
                ("1. Substrate vs Deposit Alloy Selection",
                 "• Single Alloy: Identical base plate and deposit metal (e.g. SS316L on SS316L).\n"
                 "• Dissimilar Alloys: Structural steel substrate (e.g. S355 / ER70S-6) with corrosion-resistant deposit (e.g. SS308L / SS316L / Inconel 625).\n"
                 "• Multi-Material Zones: Graded functional transitions across layers (e.g. Zone 1: Inconel 625, Zone 2: SS316L)."),
                ("2. Interfacial Thermal Expansion Mismatch & Residual Stress",
                 "• CTE Difference: Δα = α_dep - α_sub.\n"
                 "• Interfacial Shear Stress: σ_int ≈ [E_dep / (1 - ν)] · Δα · (T_interpass - T_amb).\n"
                 "• When depositing austenitic stainless steel (α ≈ 18e-6 K⁻¹) on carbon steel (α ≈ 12e-6 K⁻¹), large tensile stresses develop at the fusion boundary, promoting interfacial shear and potential debonding."),
                ("3. Constitutional Metallurgy & Ferrite Number (WRC-1992)",
                 "• Chromium Equivalent: Cr_eq = Cr + Mo + 1.5·Si + 0.5·Nb.\n"
                 "• Nickel Equivalent: Ni_eq = Ni + 30·C + 30·N + 0.5·Mn.\n"
                 "• Primary Ferrite Solidification: Target Ferrite Number (FN = 4 - 12) prevents solidification hot-cracking while maintaining ductility and toughness.")
            ]
        )

    def _show_properties_info(self):
        self._show_info_dialog(
            "Thermophysical Properties & Boundary Cooling Guide",
            "Temperature-Dependent Properties & Surface Heat Losses Guide",
            [
                ("1. Non-Linear Thermophysical Properties (k, Cp, ρ)",
                 "• Thermal Conductivity k(T): Decreases or increases with temperature; molten metal features higher effective conductivity due to fluid convection.\n"
                 "• Specific Heat Capacity Cp(T): Increases with temperature; incorporates phase transformation enthalpy."),
                ("2. Latent Heat of Fusion & Apparent Heat Capacity Method",
                 "• Solidus & Liquidus: Mushy zone bounded by T_sol and T_liq (e.g. 1402 K and 1457 K for SS316L).\n"
                 "• Enthalpy Integration: ∫[Tsol, Tliq] Cp_eff(T) dT = Cp_base·ΔT + L_f.\n"
                 "• Latent Heat of Fusion L_f: Absorbed during melting, released during solidification, smoothing thermal shock and extending cooling dwell."),
                ("3. Convective & Radiative Surface Losses",
                 "• Convection: q''_conv = h_conv · (T - T_amb) [W/m²]. Typical h_conv = 10 - 25 W/(m²·K).\n"
                 "• Stefan-Boltzmann Radiation: q''_rad = ε · σ_SB · (T⁴ - T_amb⁴) [W/m²], where σ_SB = 5.67e-8 W/(m²·K⁴).\n"
                 "• Emissivity (ε): Typically 0.40 - 0.70 for oxidised metallic surfaces at elevated temperatures. Radiation dominates heat loss above 800 °C.")
            ]
        )

    def _show_mesh_info(self):
        self._show_info_dialog(
            "Mesh Discretization & CAD Slicing Guide",
            "Finite Element Discretization & Conforming Interface Guide",
            [
                ("1. Spatial Resolution in Deposition Zone vs Substrate",
                 "• Melt Pool / HAZ Resolution: Minimum 4 elements across bead width (ny_dep ≥ 4) and at least 1-2 elements per deposited layer (nz_dep ≥ N_layer).\n"
                 "• Substrate Grading: Coarser element discretization in substrate regions far from the fusion line reduces total degrees of freedom (DOF) without compromising accuracy."),
                ("2. Conforming Delaunay Tetrahedral Meshes",
                 "• Shared Interface Nodes: Substrate top surface and Layer 1 bottom surface share identical nodes, guaranteeing displacement continuity u and heat flux conservation across the interface.\n"
                 "• Aspect Ratio Constraint: Element aspect ratios are kept below 5:1 to avoid numerical ill-conditioning in the stiffness matrix K."),
                ("3. Physics-Adaptive Auto-Meshing Engine",
                 "• Measures source dimensions (af, b, c) and adjusts element sizes to guarantee the heat source resolves across at least 3-4 element diameters.\n"
                 "• Prevents spatial source aliasing and unphysical peak temperature spikes.")
            ]
        )

    def _show_interface_info(self):
        self._show_info_dialog(
            "Conforming Deposit/Substrate Interface Guide",
            "Interface Continuity & Boundary Surface Labels Guide",
            [
                ("1. Conforming Interface Topology",
                 "• Deposit and substrate mesh partitions are generated with conforming Delaunay triangulation.\n"
                 "• Eliminates the need for artificial contact penalties or Lagrange multipliers, providing exact traction and heat flux continuity."),
                ("2. Boundary Surface Labeling",
                 "• Label 1: Substrate underside (Z = 0) — carries fixture spring supports and conductive bottom losses.\n"
                 "• Labels 2 & 4: Substrate X-minimum and X-maximum lateral faces.\n"
                 "• Labels 3 & 6: Substrate Y-minimum and Y-maximum lateral faces.\n"
                 "• Label 5: Exposed top surfaces of substrate and newly deposited weld bead — subject to convection and radiation losses.")
            ]
        )

    def _show_physics_info(self):
        self._show_info_dialog(
            "Multi-Physics Module Coupling Guide",
            "Governing Thermo-Mechanical-Metallurgical Field Equations Guide",
            [
                ("1. Transient Thermal Conduction Field",
                 "• Equation: ρ·Cp·(∂T/∂t) = ∇·(k∇T) + q_vol(x,y,z,t).\n"
                 "• Evaluates temperature history T(x,y,z,t) across active elements activated via element birth-and-death."),
                ("2. Non-Linear Thermo-Elasto-Plastic Mechanics",
                 "• Equilibrium: ∇·σ + b = 0, where b = -ρ·g·k̂ represents gravitational body force.\n"
                 "• Total Strain Decomposition: ε = ε_e + ε_p + ε_th.\n"
                 "• Thermal Strain: ε_th = α(T)·(T - T_ref)·I.\n"
                 "• J2 Associative Plasticity: von Mises yield surface f = σ_eq - σ_y(T) ≤ 0 with isotropic hardening."),
                ("3. Annealing & Stress Relaxation at Fusion Temperatures",
                 "• Molten metal above solidus (T > T_sol) loses plastic strain history (ε_p is reset to zero).\n"
                 "• Liquid elements are assigned a bounded compliant stiffness to prevent matrix singularity while transmitting zero unphysical residual stress.")
            ]
        )

    def _show_thermal_info(self):
        self._show_info_dialog(
            "Thermal Boundary Conditions & Cooling Guide",
            "Inter-Pass Cooling Strategies & Thermal Management Guide",
            [
                ("1. Inter-Pass Dwell Modes",
                 "• Fixed Time Dwell (t_wait): Torch pauses for a constant duration (e.g. 30 - 90 s) after completing each pass.\n"
                 "• Temperature-Controlled Dwell (T_interpass): Torch halts until peak monitored temperature drops below a target threshold (e.g. 150 °C).\n"
                 "• Prevents continuous heat accumulation in tall walls, stabilizing bead geometry and refining grain structure."),
                ("2. Boundary Heat Transfer Coefficients",
                 "• Free Convection: h_conv ≈ 15 W/(m²·K) across vertical and horizontal exposed faces.\n"
                 "• Backing Plate Conduction: High bottom transfer coefficient h_bot (50 - 200 W/(m²·K)) simulates intimate contact with heavy steel/copper welding tables.")
            ]
        )

    def _show_mechanical_info(self):
        self._show_info_dialog(
            "Mechanical Boundary & Fixturing Guide",
            "Fixturing Scheme, Spring Supports & Unclamp Springback Guide",
            [
                ("1. Fixturing Clamping Models",
                 "• Isostatic 3-2-1 Kinematic Support: Restrains rigid-body translations and rotations without imposing artificial thermal over-constraint.\n"
                 "• Distributed Elastic Fixture Springs: Applies distributed vertical spring stiffness Kz (N/m³) and lateral restraint Kxy to the substrate underside, modeling real-world toggle clamps."),
                ("2. Unclamp & Springback Equilibrium Solve",
                 "• During deposition: Clamps remain active (clampActive = 1), restraining bowing distortion.\n"
                 "• Post-cooling: Clamps are released (clampActive = 0) and a secondary non-linear equilibrium solve is performed.\n"
                 "• Yields true unconstrained part distortion, warpage, and permanent residual stress distribution."),
                ("3. Newton-Raphson Solver Controls",
                 "• Maximum equilibrium iterations (typical: 15 - 30).\n"
                 "• Non-linear residual force tolerance (< 1e-4) guarantees strict mechanical force balance.")
            ]
        )

    def _show_solver_info(self):
        self._show_info_dialog(
            "FreeFEM++ Time Integration & Solver Settings Guide",
            "Numerical Discretization, Timestepping & Solver Architecture Guide",
            [
                ("1. Temporal Discretization Scheme (θ-method)",
                 "• θ = 1.0 (Fully Implicit Euler): Unconditionally stable, optimal for non-linear latent heat phase transitions and steep thermal gradients.\n"
                 "• θ = 0.5 (Crank-Nicolson): Second-order accurate in time, suitable when timesteps are tightly resolved."),
                ("2. Multi-Stage Timestepping Strategy",
                 "• Deposition Timestep (Δt_weld): Typically 0.2 - 0.5 s, resolving torch travel speed across finite element lengths.\n"
                 "• Inter-Pass Timestep (Δt_dwell): Typically 2.0 - 10.0 s, capturing cooling decay between layers.\n"
                 "• Final Cooling Timestep (Δt_cool): Progressively larger increments (30 - 120 s) for long cooldown to room temperature."),
                ("3. Adaptive Time Stepping (Fourier Controller)",
                 "• Automatically computes dt based on Fourier diffusion number Fo = α·dt / h_min².\n"
                 "• Enforces Courant-like resolution across the heat source diameter while accelerating dwell and cool phases."),
                ("4. State Checkpointing & Restart",
                 "• Checkpointing can be scheduled per layer or every N steps.\n"
                 "• Saves temperature, plastic strain peeq, stress, and displacement fields to disk, enabling seamless simulation resumption.")
            ]
        )

    def _show_history_output_info(self):
        self._show_info_dialog(
            "History Output (Probe Points & Paths) Guide",
            "Continuous Time-History Extraction & Spatial Profile Probing Guide",
            [
                ("1. Purpose & Abaqus *OUTPUT, HISTORY Equivalence",
                 "• Samples field variables (T, σ_von_mises, σ_xx, σ_yy, σ_zz, ε_p, uz) directly inside the FreeFEM++ finite element solver at true simulation time steps.\n"
                 "• Eliminates post-processing interpolation errors and generates publication-grade 2D time-history plots."),
                ("2. Geometry-Linked Baseplate Center Stations (Xc = Lx/2, Yc = W_sub/2)",
                 "• Station 1 (Z = 0.50 · H_sub): Substrate mid-depth — captures baseplate conductive heat soak and longitudinal bending moment.\n"
                 "• Station 2 (Z = 1.00 · H_sub): Substrate surface / fusion boundary — tracks peak interfacial thermal shock and maximum shear stress τ_xz.\n"
                 "• Station 3 (Z = H_sub + 0.20 · H_build): Lower wall build — evaluates remelting and heat accumulation in early layers.\n"
                 "• Station 4 (Z = H_sub + 0.50 · H_build): Mid wall build — measures quasi-steady-state deposition thermal cycles and cooling rate Δt_8/5.\n"
                 "• Station 5 (Z = H_sub + 1.00 · H_build): Top wall build — captures final layer cooling and free-surface thermal shrinkage."),
                ("3. Spatial Probe Paths (Linear Profile Extraction)",
                 "• Format: x1, y1, z1, x2, y2, z2 (one per line, in mm).\n"
                 "• Default Path: Longitudinal centerline along substrate mid-depth (0, Yc, H_sub/2) → (Lx, Yc, H_sub/2).\n"
                 "• Generates spatial profiles of residual stress and warpage along the weld direction."),
                ("4. Dynamic Auto-Sync with Geometry Window",
                 "• Any modification to baseplate dimensions (Lx, Ly_base, hz_base) or layer build (N_layer, h_bead, remelt_ratio) immediately recalculates probe coordinates.\n"
                 "• Manual edits are preserved and can be restored at any time by clicking 'Reset to Standard Probes'.")
            ]
        )

    def _show_openfoam_info(self):
        self._show_info_dialog(
            "OpenFOAM 2412 Melt Pool Hydrodynamics Guide",
            "beamWeldFoam VOF Fluid Flow & Marangoni Convection Guide",
            [
                ("1. Multiphase Volume of Fluid (VOF) Free Surface Tracking",
                 "• Solves Navier-Stokes equations with phase fraction α (1 = liquid metal, 0 = shielding gas).\n"
                 "• Captures dynamic melt pool geometry, surface depression, droplet impingement, and liquid surface ripple formation."),
                ("2. Marangoni Thermocapillary Convection",
                 "• Surface Shear Stress: τ = (∂γ/∂T) · ∇_s T.\n"
                 "• Outward Flow (∂γ/∂T < 0): Standard pure alloys produce outward fluid circulation, yielding wide, shallow melt pools.\n"
                 "• Inward Flow (∂γ/∂T > 0): Presence of surface-active elements (e.g. sulfur/oxygen) inverts surface tension gradient, driving inward convection and deep finger penetration."),
                ("3. Darcy-Type Porous Flow Damping in Mushy Zone",
                 "• Momentum Sink: S = -[C_mushy · (1 - f_L)² / (f_L³ + ε)] · u.\n"
                 "• Smoothly extinguishes fluid velocity as metal cools through the solidus-liquidus mushy zone.")
            ]
        )

    def _show_ultrasonic_info(self):
        self._show_info_dialog(
            "Ultrasonic-Assisted WAAM (UA-WAAM) Guide",
            "Acoustic Cavitation, Streaming & Grain Refinement Guide",
            [
                ("1. Ultrasonic High-Frequency Acoustic Energy (20 - 40 kHz)",
                 "• Transmits high-intensity acoustic vibration directly into the substrate or molten weld pool during deposition.\n"
                 "• Acoustic power input P_us and horn vibration amplitude A_us induce dynamic alternating stress waves."),
                ("2. Acoustic Cavitation & Equiaxed Grain Refinement",
                 "• Cavitation Bubble Collapse: Generates localized shockwaves (> 1 GPa) and micro-jets that shatter growing columnar dendrite arms.\n"
                 "• Columnar-to-Equiaxed Transition (CET): Promotes heterogeneous nucleation, transforming coarse anisotropic columnar grains into fine, isotropic equiaxed microstructure."),
                ("3. Acoustic Softening & Residual Stress Mitigation",
                 "• Oscillatory stress waves assist dislocation motion past pinning obstacles (Blaha effect).\n"
                 "• Drastically reduces macroscopic yield stress during deposition and relaxes peak residual tensile stresses by 30 - 50%.")
            ]
        )

    def _show_microstructure_info(self):
        self._show_info_dialog(
            "Microstructure Kinetics & Hardness Guide",
            "Solidification Morphology, Phase Transformations & HV Mapping Guide",
            [
                ("1. Solidification Parameters (G, R, Cooling Rate)",
                 "• Thermal Gradient G = |∇T| [K/mm].\n"
                 "• Solidification Growth Velocity R = v_scan · cos(θ) [mm/s].\n"
                 "• Cooling Rate: dT/dt = G · R [K/s].\n"
                 "• G/R Ratio dictates solidification mode (planar → cellular → columnar dendritic → equiaxed)."),
                ("2. Secondary Dendrite Arm Spacing (SDAS)",
                 "• Empirical Power Law: λ_2 = A · (G · R)^(-n).\n"
                 "• Higher cooling rates produce significantly finer dendrite arm spacing, directly improving yield strength and ductility."),
                ("3. Solid-State Phase Transformation Kinetics & Vickers Hardness",
                 "• Austenite decomposition into ferrite, pearlite, bainite, and martensite modeled via Johnson-Mehl-Avrami-Kolmogorov (JMAK) kinetics.\n"
                 "• Vickers Hardness (HV): Evaluated across the build via rule-of-mixtures based on local phase fractions.")
            ]
        )

    def _show_solve_cadence_info(self):
        """Display an engineering guide explaining structural solve cadence and trade-offs."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Structural Solve Cadence & Equilibrium Guide")
        dlg.geometry("760x640")
        dlg.minsize(680, 500)
        dlg.transient(self.root)
        dlg.grab_set()

        canvas = tk.Canvas(dlg, highlightthickness=0, bg=self.colors["bg_card"])
        scroll = ttk.Scrollbar(dlg, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas, bg=self.colors["bg_card"], padx=18, pady=14)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        cw = canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        title = tk.Label(
            frame, text="Structural Solve Cadence & Equilibrium Guide",
            font=FONT_TITLE, fg=self.colors["accent"], bg=self.colors["bg_card"]
        )
        title.pack(anchor=tk.W, pady=(0, 8))

        intro_text = (
            "Solve Cadence defines the coupling frequency between the transient thermal "
            "conduction field T(x, y, z, t) and the non-linear thermo-elasto-plastic mechanical "
            "equilibrium solver K·u = F_th.\n\n"
            "Because non-linear J₂ plasticity with isotropic hardening requires iterative "
            "Newton-Raphson equilibrium iterations at every evaluation, solving mechanics at "
            "every thermal time step can be computationally demanding. The simulator "
            "provides three calibrated cadence strategies:"
        )
        tk.Label(
            frame, text=intro_text, font=FONT_LABEL, fg=self.colors["fg"],
            bg=self.colors["bg_card"], justify=tk.LEFT, wraplength=690
        ).pack(anchor=tk.W, pady=(0, 10))

        cadences = [
            ("1. 'interpass' (Inter-Pass Cadence — Once per Layer)",
             "• Mechanism: Structural equilibrium is evaluated once at the completion of each deposit layer.\n"
             "• Performance: 5x to 15x faster than continuous solve.\n"
             "• Best For: Large multi-layer walls, fast macro-distortion predictions, overall substrate warpage, "
             "and final unconstrained springback estimation.\n"
             "• Physics Assumption: Elastic-plastic strain accumulates during high-speed bead travel and relaxes at dwell."),

            ("2. 'layer_n' (Every N Weld Steps Cadence)",
             "• Mechanism: Structural mechanics runs periodically every N thermal time increments (e.g. N = 5 steps).\n"
             "• Performance: 2x to 5x faster with minimal loss of accuracy.\n"
             "• Best For: Capturing transient thermal shock waves and gradual bending accumulation during long weld passes "
             "without paying the penalty of full step-by-step solving."),

            ("3. 'weld' (Continuous Every Step — Abaqus / ANSYS Benchmark)",
             "• Mechanism: Structural equilibrium is solved at EVERY single deposition time increment Δt_weld.\n"
             "• Performance: Computationally demanding (1x baseline).\n"
             "• Best For: Rigorous academic research, thin-wall hot cracking assessment, transient stress reversal, "
             "and high-fidelity experimental strain gauge validation.")
        ]

        for head, body in cadences:
            box = tk.LabelFrame(frame, text=f"  {head}  ", font=FONT_SECTION, bg=self.colors["bg_card"],
                                fg=self.colors["accent2"], padx=12, pady=8)
            box.pack(fill=tk.X, pady=6)
            tk.Label(box, text=body, font=FONT_LABEL, fg=self.colors["fg"],
                     bg=self.colors["bg_card"], justify=tk.LEFT, wraplength=670).pack(anchor=tk.W)

        btn_close = tk.Button(
            frame, text="Close Guide", font=FONT_BTN, bg=self.colors["button"],
            fg=self.colors["button_fg"], command=dlg.destroy, padx=14, pady=5,
            relief=tk.FLAT, bd=0
        )
        btn_close.pack(anchor=tk.E, pady=(12, 4))

    def _get_default_probe_points_text(self):
        """
        Compute standard 5 physical history probe stations at baseplate center (X=Lx/2, Y=Ly/2):
        Station 1: Substrate Mid-Depth (Z = H_sub / 2)
        Station 2: Substrate Surface / Fusion Interface (Z = H_sub)
        Station 3: 20% Wall Build Height (Z = H_sub + 0.20 * H_build)
        Station 4: 50% Wall Build Height (Z = H_sub + 0.50 * H_build)
        Station 5: 100% Wall Build Height (Z = H_sub + 1.00 * H_build)
        """
        try:
            lx_mm = self._f(self._param_vars.get("lx", (None,))[0], 150.0) if hasattr(self, "_param_vars") and "lx" in self._param_vars else 150.0
            ly_mm = self._f(self._param_vars.get("ly_base", (None,))[0], 150.0) if hasattr(self, "_param_vars") and "ly_base" in self._param_vars else 150.0
            hz_sub_mm = self._f(self._param_vars.get("hz_base", (None,))[0], 10.0) if hasattr(self, "_param_vars") and "hz_base" in self._param_vars else 10.0
            
            xc = lx_mm / 2.0
            yc = ly_mm / 2.0
            
            if hasattr(self, "_effective_layer_count"):
                _, _, h_build_mm = self._effective_layer_count()
            else:
                h_build_mm = self._current_build_height_mm() if hasattr(self, "_current_build_height_mm") else 20.0
            
            z1 = hz_sub_mm / 2.0
            z2 = hz_sub_mm
            z3 = hz_sub_mm + 0.20 * h_build_mm
            z4 = hz_sub_mm + 0.50 * h_build_mm
            z5 = hz_sub_mm + 1.00 * h_build_mm
            
            lines = [
                f"{xc:.1f}, {yc:.1f}, {z1:.2f}   # Station 1: Substrate Mid-Depth (0.5 * H_sub)",
                f"{xc:.1f}, {yc:.1f}, {z2:.2f}   # Station 2: Substrate Top Surface / Interface (H_sub)",
                f"{xc:.1f}, {yc:.1f}, {z3:.2f}   # Station 3: Lower Wall Build (H_sub + 0.20 * H_build)",
                f"{xc:.1f}, {yc:.1f}, {z4:.2f}   # Station 4: Mid Wall Build (H_sub + 0.50 * H_build)",
                f"{xc:.1f}, {yc:.1f}, {z5:.2f}   # Station 5: Top Wall Build (H_sub + 1.00 * H_build)",
            ]
            return "\n".join(lines)
        except Exception:
            return (
                "75.0, 75.0, 2.50   # Station 1: Substrate Mid-Depth\n"
                "75.0, 75.0, 5.00   # Station 2: Substrate Surface\n"
                "75.0, 75.0, 7.64   # Station 3: 20% Build Height\n"
                "75.0, 75.0, 11.60  # Station 4: 50% Build Height\n"
                "75.0, 75.0, 18.20  # Station 5: 100% Build Height"
            )

    def _get_default_probe_paths_text(self):
        """
        Compute standard longitudinal centerline probe path at baseplate mid-depth:
        (x1=0, y1=W_sub/2, z1=H_sub/2) -> (x2=Lx, y2=W_sub/2, z2=H_sub/2)
        """
        try:
            lx_mm = self._f(self._param_vars.get("lx", (None,))[0], 150.0) if hasattr(self, "_param_vars") and "lx" in self._param_vars else 150.0
            ly_mm = self._f(self._param_vars.get("ly_base", (None,))[0], 150.0) if hasattr(self, "_param_vars") and "ly_base" in self._param_vars else 150.0
            hz_sub_mm = self._f(self._param_vars.get("hz_base", (None,))[0], 10.0) if hasattr(self, "_param_vars") and "hz_base" in self._param_vars else 10.0
            yc = ly_mm / 2.0
            zc = hz_sub_mm / 2.0
            return f"0.0, {yc:.1f}, {zc:.2f}, {lx_mm:.1f}, {yc:.1f}, {zc:.2f}   # Longitudinal centerline at substrate mid-depth"
        except Exception:
            return "0.0, 75.0, 5.0, 150.0, 75.0, 5.0"

    def _reset_default_probe_points(self):
        """Reset probe points and paths to standard baseplate center and geometry-linked elevation stations."""
        self._user_custom_probes = False
        self._user_custom_paths = False
        if hasattr(self, "probe_points_text"):
            default_txt = self._get_default_probe_points_text()
            self.probe_points_text.delete("1.0", tk.END)
            self.probe_points_text.insert("1.0", default_txt)
        if hasattr(self, "probe_paths_text"):
            path_txt = self._get_default_probe_paths_text()
            self.probe_paths_text.delete("1.0", tk.END)
            self.probe_paths_text.insert("1.0", path_txt)
        self._update_probe_geom_badge()

    def _on_probe_points_user_key(self, event=None):
        self._user_custom_probes = True
        self._update_probe_geom_badge()

    def _on_probe_paths_user_key(self, event=None):
        self._user_custom_paths = True
        self._update_probe_geom_badge()

    def _update_probe_geom_badge(self):
        if not hasattr(self, "probe_geom_badge_lbl"):
            return
        is_custom = getattr(self, "_user_custom_probes", False) or getattr(self, "_user_custom_paths", False)
        if is_custom:
            self.probe_geom_badge_lbl.config(
                text="⚠️ Custom Probe Coordinates Active (click 'Reset' below to re-link to Geometry)",
                fg="#d97706"
            )
        else:
            try:
                lx_mm = self._f(self._param_vars.get("lx", (None,))[0], 150.0) if hasattr(self, "_param_vars") and "lx" in self._param_vars else 150.0
                ly_mm = self._f(self._param_vars.get("ly_base", (None,))[0], 150.0) if hasattr(self, "_param_vars") and "ly_base" in self._param_vars else 150.0
                hz_sub_mm = self._f(self._param_vars.get("hz_base", (None,))[0], 10.0) if hasattr(self, "_param_vars") and "hz_base" in self._param_vars else 10.0
                if hasattr(self, "_effective_layer_count"):
                    _, _, h_build_mm = self._effective_layer_count()
                else:
                    h_build_mm = 20.0
                xc = lx_mm / 2.0
                yc = ly_mm / 2.0
                self.probe_geom_badge_lbl.config(
                    text=f"✓ Auto-Linked to Geometry: Center ({xc:.1f}, {yc:.1f}) mm | H_sub={hz_sub_mm:.1f} mm | H_build={h_build_mm:.2f} mm",
                    fg="#2563eb"
                )
            except Exception:
                self.probe_geom_badge_lbl.config(
                    text="✓ Auto-Linked to Geometry Window Coordinates",
                    fg="#2563eb"
                )

    def _build_solver_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Solver")
        self._reg(tab, "card_frame")

        lf1 = tk.LabelFrame(tab, text="Temporal Discretization", font=FONT_SECTION, padx=8, pady=4)
        lf1.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf1, "card_labelframe")
        self._add_info_btn_to_labelframe(lf1, self._show_solver_info)
        self._make_param_row(lf1, "Time Scheme Weight (θ):", "theta", 0, "[-]")
        self._make_param_row(lf1, "Deposition Timestep (Δt_weld):", "dt_weld", 1, "s")
        self._make_param_row(lf1, "Inter-Pass Timestep (Δt_dwell):", "dt_wait", 2, "s")
        self._make_param_row(lf1, "Final Cooling Timestep (Δt_cool):", "dt_cool", 3, "s")
        self._make_param_row(lf1, "Cooling Steps Count (N_cool):", "n_cool_steps", 4, "steps", type_="int")

        # Adaptive time stepping. Off by default (fixed schedules, legacy
        # behaviour). When enabled, dt is derived from the diffusion (Fourier)
        # number and the measured minimum element size, so thermal accuracy no
        # longer depends on the user's Delta-t guess.
        lf_ada = tk.LabelFrame(tab, text="Adaptive Time Stepping", font=FONT_SECTION,
                               padx=8, pady=4)
        lf_ada.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_ada, "card_labelframe")
        ttk.Label(lf_ada, text="Controller:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        self.adaptive_dt_var = tk.StringVar(value="off")
        ada_combo = ttk.Combobox(lf_ada, textvariable=self.adaptive_dt_var,
                                 state="readonly", width=22, values=["off", "fourier"])
        ada_combo.grid(row=0, column=1, padx=4, pady=2, sticky=tk.W)
        ttk.Label(lf_ada, text="fourier = limit Fo = α·Δt/h² from measured h_min "
                               "(dt also capped so the arc resolves its source width)",
                  foreground="#64748b").grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=4)
        self._make_param_row(lf_ada, "Target Diffusion Number (Fo):", "adaptive_fo_target", 2, "[-]")
        self._make_param_row(lf_ada, "Minimum Δt:", "adaptive_dt_min", 3, "s")
        self._make_param_row(lf_ada, "Maximum Δt:", "adaptive_dt_max", 4, "s")

        # --- History output (Abaqus *OUTPUT,HISTORY equivalent) ---
        # The solver samples these points/paths itself and writes CSV rows tagged
        # with the true simulated time, so nothing downstream has to guess where
        # to sample or invent a time axis.
        lf_hist = tk.LabelFrame(tab, text="History Output (Probe Points & Paths)",
                                font=FONT_SECTION, padx=8, pady=4)
        lf_hist.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_hist, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_hist, self._show_history_output_info)

        # Dynamic geometry readout badge
        self.probe_geom_badge_lbl = tk.Label(
            lf_hist, text="✓ Auto-Linked to Geometry Window Coordinates",
            font=FONT_SMALL_BOLD, fg="#2563eb", bg=self.colors.get("bg_card", "#ffffff")
        )
        self.probe_geom_badge_lbl.grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=4, pady=(2, 6))

        ttk.Label(lf_hist, text="Probe points (mm, one per line, x,y,z):").grid(
            row=1, column=0, sticky=tk.NW, padx=4, pady=2)
        self.probe_points_text = tk.Text(lf_hist, height=6, width=44, font=FONT_MONO)
        self.probe_points_text.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=4)
        self.probe_points_text.insert("1.0", self._get_default_probe_points_text())
        self.probe_points_text.bind("<Key>", self._on_probe_points_user_key)

        btn_reset_probes = tk.Button(
            lf_hist, text="🔄 Reset to Standard Probes & Paths (Baseplate Center & Geometry Drivers)",
            font=FONT_SMALL, relief=tk.FLAT, padx=6, pady=2, cursor="hand2",
            command=self._reset_default_probe_points
        )
        btn_reset_probes.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=4, pady=(2, 4))
        self._reg(btn_reset_probes, "button")

        ttk.Label(lf_hist, text="Probe paths (mm; x1,y1,z1,x2,y2,z2 per line):").grid(
            row=3, column=0, sticky=tk.NW, padx=4, pady=2)
        self.probe_paths_text = tk.Text(lf_hist, height=3, width=44, font=FONT_MONO)
        self.probe_paths_text.grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=4)
        self.probe_paths_text.insert("1.0", self._get_default_probe_paths_text())
        self.probe_paths_text.bind("<Key>", self._on_probe_paths_user_key)

        self._make_param_row(lf_hist, "Stations per Path:", "path_samples", 4, "[-]", type_="int")
        self._make_param_row(lf_hist, "Write History Every N Steps:", "probe_interval", 5, "steps", type_="int")
        self._update_probe_geom_badge()

        # --- Restart / checkpoint ---
        lf_ck = tk.LabelFrame(tab, text="Restart / Checkpoint", font=FONT_SECTION,
                              padx=8, pady=4)
        lf_ck.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_ck, "card_labelframe")
        self._add_info_btn_to_labelframe(lf_ck, self._show_solver_info)
        ttk.Label(lf_ck, text="Checkpointing:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        self.restart_mode_var = tk.StringVar(value="off")
        ck_combo = ttk.Combobox(lf_ck, textvariable=self.restart_mode_var,
                                state="readonly", width=18,
                                values=["off", "layer", "step"])
        ck_combo.grid(row=0, column=1, padx=4, pady=2, sticky=tk.W)
        ttk.Label(lf_ck, text="layer = checkpoint after each layer (resume unit); "
                              "step = every N steps",
                  foreground="#64748b").grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=4)
        self._make_param_row(lf_ck, "Steps Between Checkpoints:", "restart_every", 2, "steps", type_="int")
        self.restart_resume_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf_ck, text="Resume from existing checkpoint if present",
                        variable=self.restart_resume_var).grid(
            row=3, column=0, columnspan=3, sticky=tk.W, padx=4, pady=2)

        lf2 = tk.LabelFrame(tab, text="Solver & Output Config", font=FONT_SECTION, padx=8, pady=4)
        lf2.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf2, "card_labelframe")
        self._add_info_btn_to_labelframe(lf2, self._show_solver_info)
        self._make_param_row(lf2, "Output Directory:", "output_dir", 0, width=25, type_="str")
        self._make_param_row(lf2, "Save Every N Weld Steps:", "save_every_n_weld", 1, "", type_="int")
        self._make_param_row(lf2, "Save Every N Cool Steps:", "save_every_n_cool", 2, "", type_="int")

    # ============================================================
    # ACTION HANDLERS
    # ============================================================

    def _collect_params(self) -> dict:
        params = EDPBuilder.default_params()
        for key, (var, type_) in self._param_vars.items():
            try:
                val_str = var.get().strip()
                if not val_str or val_str.lower() in ("none", "null", "auto"):
                    params[key] = None
                    continue
                if type_ == "str":
                    params[key] = val_str
                elif type_ == "int":
                    params[key] = int(float(val_str))
                else:
                    params[key] = float(val_str)
                # Entry parsed OK: allow future warnings if it breaks again.
                warned = getattr(self, "_warned_invalid_keys", None)
                if warned is not None:
                    warned.discard(key)
            except Exception as e:
                # Warn once per key per session to avoid console spam
                # (_collect_params runs on every UI refresh / mesh calc).
                warned = getattr(self, "_warned_invalid_keys", None)
                if warned is None:
                    warned = set()
                    self._warned_invalid_keys = warned
                if key not in warned:
                    warned.add(key)
                    try:
                        self._log(f"[PARAM] [WARNING] Invalid entry for '{key}': {e}. Keeping {params.get(key)}.\n")
                    except Exception:
                        pass

        # Convert user-friendly millimeter entries into SI meters for FreeFEM
        # solver. The rule lives in mesh/units.py (one plausibility table for
        # the GUI, the builder and the OpenFOAM coupling): a value larger than
        # any legitimate SI value for that quantity is millimetres.
        # Conversion runs silently during routine UI interaction and refresh.
        try:
            from mesh.units import dim_to_m
        except Exception:
            dim_to_m = None

        def _convert(key, kind):
            if key not in params or params[key] is None or dim_to_m is None:
                return
            try:
                before = float(params[key])
            except (ValueError, TypeError):
                return
            try:
                after = dim_to_m(params[key], key)
            except ValueError:
                return
            if after != before:
                params[key] = after

        for _k in MM_DIMENSION_KEYS:
            _convert(_k, "millimetres")
        for _k in MM_SPEED_KEYS:
            _convert(_k, "mm/s")

        params["heat_source"] = self.source_var.get()
        if params["heat_source"] in ("p48_surface", "p48_gmaw", "p48"):
            for _k in ("af", "ar", "bG", "cG", "ff", "fr"):
                params[_k] = None
            if "eta_drop" not in params or params["eta_drop"] is None:
                params["eta_drop"] = 0.25
            if "T_drop" not in params or params["T_drop"] is None:
                params["T_drop"] = 1627.0
        elif params["heat_source"] != "goldak":
            for _k in ("af", "ar", "bG", "cG", "ff", "fr"):
                params[_k] = None
        params["material_base"] = self.material_base_var.get()
        params["material_dep"] = self.material_dep_var.get()
        params["material"] = self.material_dep_var.get()
        params["mesh_source"] = self.mesh_source_var.get()
        params["mesh_file"] = self.mesh_file_var.get()
        stl_p = getattr(self, "loaded_stl_path", "")
        if not stl_p and params["mesh_file"] and params["mesh_file"].lower().endswith((".stl", ".ply", ".obj")):
            stl_p = params["mesh_file"]
        params["stl_file"] = stl_p
        params["enable_radiation"] = self.rad_var.get()
        params["enable_latent"] = self.latent_var.get()
        params["enable_mechanical"] = self.mech_var.get()
        params["enable_microstructure"] = self.micro_var.get()
        params["enable_p48_buoyancy"] = getattr(self, "p48_buoyancy_var", tk.BooleanVar(value=True)).get()
        params["enable_p48_droplets"] = getattr(self, "p48_droplet_var", tk.BooleanVar(value=True)).get()
        params["enable_openfoam"] = self.foam_var.get()
        params["enable_ultrasonic"] = getattr(self, "enable_ultrasonic_var", tk.BooleanVar(value=False)).get()

        # --- UVA physics parameters ---
        if hasattr(self, "us_freq_var"):
            try:
                params["ultrasonic_frequency"] = float(self.us_freq_var.get())
            except Exception:
                params["ultrasonic_frequency"] = 20000.0
        if hasattr(self, "us_amp_var"):
            try:
                params["ultrasonic_amplitude"] = float(self.us_amp_var.get()) * 1.0e-6
            except Exception:
                params["ultrasonic_amplitude"] = 15.0e-6
        if hasattr(self, "us_coupling_var"):
            try:
                params["ultrasonic_coupling_eff"] = float(self.us_coupling_var.get())
            except Exception:
                params["ultrasonic_coupling_eff"] = 0.15
        if hasattr(self, "us_attenuation_var"):
            try:
                params["ultrasonic_attenuation"] = float(self.us_attenuation_var.get())
            except Exception:
                params["ultrasonic_attenuation"] = 0.12
        if hasattr(self, "us_csound_var"):
            try:
                params["ultrasonic_c_sound"] = float(self.us_csound_var.get())
            except Exception:
                params["ultrasonic_c_sound"] = 5790.0
        if hasattr(self, "us_kstream_var"):
            try:
                params["ultrasonic_k_stream"] = float(self.us_kstream_var.get())
            except Exception:
                params["ultrasonic_k_stream"] = 45.0
        if hasattr(self, "us_cavthresh_var"):
            try:
                params["ultrasonic_cav_threshold"] = float(self.us_cavthresh_var.get()) * 1.0e6
            except Exception:
                params["ultrasonic_cav_threshold"] = 1.0e6
        if hasattr(self, "us_streaming_var"):
            params["ultrasonic_streaming"] = bool(self.us_streaming_var.get())
        if hasattr(self, "us_heatgen_var"):
            params["ultrasonic_heat_gen"] = bool(self.us_heatgen_var.get())
        if hasattr(self, "us_cavitation_var"):
            params["ultrasonic_cavitation"] = bool(self.us_cavitation_var.get())
        if hasattr(self, "us_momentum_var"):
            params["ultrasonic_momentum"] = bool(self.us_momentum_var.get())

        # Wire Feedstock & Mass Deposition Aliases and Physical Quantities
        d_wire_m = params.get("d_wire", 1.2e-3)
        if d_wire_m is None or d_wire_m <= 0:
            d_wire_m = 1.2e-3
        params["d_wire"] = d_wire_m
        params["wire_diameter"] = d_wire_m
        params["wire_diameter_mm"] = d_wire_m * 1000.0

        v_wire_m_min = params.get("v_wire", 5.0)
        if v_wire_m_min is None or v_wire_m_min <= 0:
            v_wire_m_min = 5.0
        params["v_wire"] = v_wire_m_min
        params["wire_feed_speed"] = v_wire_m_min
        params["wire_feed_speed_m_min"] = v_wire_m_min
        params["wire_feed_speed_m_s"] = v_wire_m_min / 60.0

        wire_area_m2 = (math.pi * (d_wire_m ** 2)) / 4.0
        params["wire_area_m2"] = wire_area_m2
        params["wire_area_mm2"] = wire_area_m2 * 1e6

        vol_rate_m3_s = wire_area_m2 * params["wire_feed_speed_m_s"]
        params["wire_vol_rate_m3_s"] = vol_rate_m3_s

        rho_dep = float(params.get("rho", 7850.0))
        m_dep_kg_s = rho_dep * vol_rate_m3_s
        params["wire_mass_rate_kg_s"] = m_dep_kg_s
        params["wire_mass_rate_kg_h"] = m_dep_kg_s * 3600.0

        v_scan_m_s = params.get("v_scan", 0.010)
        params["a_bead_theo_m2"] = (vol_rate_m3_s / v_scan_m_s) if v_scan_m_s > 0 else 0.0
        params["a_bead_theo_mm2"] = params["a_bead_theo_m2"] * 1e6

        # --- Incremental mechanics / equilibrium controls (Solver + Physics tabs) ---
        if hasattr(self, "mech_cadence_var"):
            params["mech_solve_cadence"] = self.mech_cadence_var.get()
        if hasattr(self, "springback_var"):
            params["springback_solve"] = bool(self.springback_var.get())
            # Springback is only meaningful if the fixture was a real spring
            # clamp that we can release.
            params["dynamic_clamping"] = bool(self.springback_var.get())

        # --- Adaptive time stepping (Solver tab) ---
        if hasattr(self, "adaptive_dt_var"):
            params["adaptive_dt"] = self.adaptive_dt_var.get()

        # --- History output: probes given in mm, converted to metres ---
        def _parse_probe_block(textbox, label):
            """Parse one coordinate group per non-comment line into metres."""
            try:
                raw_lines = textbox.get("1.0", "end").splitlines()
            except Exception:
                return []
            out = []
            for ln in raw_lines:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                if "#" in ln:
                    ln = ln.split("#", 1)[0].strip()
                try:
                    vals = [float(v) for v in ln.replace(";", ",").split(",") if v.strip()]
                except ValueError:
                    print(f"[GUI] ignoring malformed {label} line: {ln!r}")
                    continue
                if len(vals) % 3 or len(vals) < 3:
                    print(f"[GUI] ignoring {label} line (need 3n coords): {ln!r}")
                    continue
                out.append([v / 1000.0 for v in vals])
            return out

        if hasattr(self, "probe_points_text"):
            pts = _parse_probe_block(self.probe_points_text, "probe point")
            
            # If no probe points were specified by the user, provide standard 5 elevation stations at plate center
            if not pts:
                try:
                    _lx = float(params.get("lx", 0.15))
                    _ly = float(params.get("ly_base", 0.15))
                    _hz = float(params.get("hz_base", 0.005))
                    _h_build = float(params.get("H_build_actual_mm", 13.20)) / 1000.0
                    xc = _lx / 2.0
                    yc = _ly / 2.0
                    pts = [
                        [xc, yc, _hz / 2.0],
                        [xc, yc, _hz],
                        [xc, yc, _hz + 0.20 * _h_build],
                        [xc, yc, _hz + 0.50 * _h_build],
                        [xc, yc, _hz + 1.00 * _h_build],
                    ]
                except Exception:
                    pass

            params["probe_points"] = pts
            if hasattr(self, "probe_paths_text"):
                paths = _parse_probe_block(self.probe_paths_text, "probe path")
                paths = [q for q in paths if len(q) >= 6]
                params["probe_paths"] = paths

        # --- Restart / checkpoint ---
        if hasattr(self, "restart_mode_var"):
            params["restart"] = self.restart_mode_var.get()
            params["restart_resume"] = bool(self.restart_resume_var.get())

        # --- Deposition Pattern ---
        if hasattr(self, "deposition_pattern_var"):
            params["deposition_pattern"] = self.deposition_pattern_var.get()
        else:
            params["deposition_pattern"] = "bidirectional"

        # --- Ultrasonic Tab Variables & Physics Module Gating ---
        is_us_active = bool(params.get("enable_ultrasonic", False))
        if is_us_active and hasattr(self, "us_freq_var"):
            try:
                params["ultrasonic_frequency"] = float(self.us_freq_var.get())
            except Exception:
                params["ultrasonic_frequency"] = 20000.0
        else:
            params["ultrasonic_frequency"] = 0.0

        # Ensure dimensioning driver (N_layer vs Target Wall Height) is strictly applied.
        # This is the SAME formula as _effective_layer_count/_sync_from_h_bead:
        # H_build drives N via ceil, so the solver can never silently disagree
        # with the Geometry tab. Any correction is written back to the N_layer
        # entry and logged (the stale "shows 8, solves 10" state is refused).
        layer_mode = getattr(self, "_layer_mode", None)
        mode_val = layer_mode.get() if layer_mode is not None else "n_layers"
        if mode_val in ("height", "h_bead"):
            try:
                nl_eff, target_h_eff, h_actual_eff = self._effective_layer_count()
            except Exception:
                nl_eff, target_h_eff, h_actual_eff = None, None, None
            if nl_eff is not None:
                try:
                    raw_entry = self._param_vars["n_layers"][0].get().strip() if "n_layers" in self._param_vars else ""
                    entry_nl = int(float(raw_entry)) if raw_entry else None
                except Exception:
                    entry_nl = None
                params["n_layers"] = max(1, int(nl_eff))
                params["H_build_target_mm"] = target_h_eff
                params["H_build_actual_mm"] = h_actual_eff
                if entry_nl != params["n_layers"] and "n_layers" in self._param_vars:
                    try:
                        self._param_vars["n_layers"][0].set(str(params["n_layers"]))
                    except Exception:
                        pass
                    try:
                        self._update_geom_fusion_label()
                    except Exception:
                        pass
                try:
                    self._log(f"[LAYERS] H_build driver: target={target_h_eff:.2f} mm "
                              f"(h1={self._f(self._param_vars.get('h_bead', (None,))[0], 4.0) if 'h_bead' in self._param_vars else 4.0:.2f} mm) "
                              f"-> N_layer={params['n_layers']} (actual wall {h_actual_eff:.2f} mm, ceil overshoot expected). "
                              f"Solver and display now agree.\n")
                except Exception:
                    pass
        else:
            try:
                raw_nl = self._param_vars["n_layers"][0].get().strip() if "n_layers" in self._param_vars else "10"
                params["n_layers"] = max(1, int(float(raw_nl)))
            except Exception:
                params["n_layers"] = 10
            try:
                _, _, h_actual_n = self._effective_layer_count()
                params["H_build_actual_mm"] = h_actual_n
                if hasattr(self, "_build_height_var"):
                    self._build_height_var.set(f"{h_actual_n:.2f}")
            except Exception:
                pass

        nl = max(int(params.get("n_layers", 10)), 1)
        params["n_layers"] = nl
        params["num_layers"] = nl
        try:
            nz_user = int(float(params.get("nz_dep", nl)))
        except (TypeError, ValueError):
            nz_user = nl
        prev_nl = getattr(self, "_last_n_layers", None)
        if prev_nl is not None and nz_user != prev_nl and nz_user != nl:
            # The user diverged nz_dep from n_layers: respect it and say what
            # it means for the source sizing below.
            if getattr(self, "_last_logged_nz_override", None) != (nz_user, nl):
                self._last_logged_nz_override = (nz_user, nl)
                try:
                    cpl = nz_user / max(nl, 1)
                    self._log(f"[MESH] nz_dep={nz_user} with n_layers={nl}: "
                              f"{cpl:.2f} cells/layer (user override kept).\n")
                except Exception:
                    pass
            params["nz_dep"] = max(1, nz_user)
        else:
            self._last_logged_nz_override = None
            params["nz_dep"] = nl
        self._last_n_layers = nl

        ty_dep = float(params.get("ty_dep", 0.005))
        params["ty_dep"] = ty_dep
        params["wBead"] = ty_dep

        # Populate synchronized remelt and bead dimensions
        remelt = float(params.get("remelt_ratio", 0.80))
        h1 = float(params.get("h_bead", params.get("h_bead_first", 0.004)))
        params["h_bead_first"] = h1
        params["h_bead"] = h1
        params["hBead"] = h1
        params["remelt_ratio"] = remelt
        params["h_bead_subseq"] = h1 * remelt
        
        # Collect dynamic multi-material zones (strictly gated by Ultrasonic physics state)
        if is_us_active and hasattr(self, "us_zone_entries") and self.us_zone_entries:
            zones_data = []
            for entry in self.us_zone_entries:
                try:
                    val_str = entry["layers_var"].get().strip()
                    lz = int(val_str) if val_str.isdigit() else 1
                except Exception:
                    lz = 1
                mz = entry["mat_var"].get().strip()
                if lz > 0:
                    zones_data.append({"layers": lz, "material": mz})
            if zones_data:
                # Ensure zone layer allocation perfectly matches total n_layers
                tot_z = sum(z["layers"] for z in zones_data)
                target_nl = params["n_layers"]
                if tot_z != target_nl and tot_z > 0:
                    scaled_zones = []
                    cum_l = 0
                    for i, z in enumerate(zones_data):
                        if i == len(zones_data) - 1:
                            l_alloc = max(1, target_nl - cum_l)
                        else:
                            l_alloc = max(1, int(round(z["layers"] * target_nl / tot_z)))
                            cum_l += l_alloc
                        scaled_zones.append({"layers": l_alloc, "material": z["material"]})
                    zones_data = scaled_zones
                params["zones"] = zones_data
                params["zonea_layers"] = zones_data[0]["layers"]
                params["zonea_material"] = zones_data[0]["material"]
                if len(zones_data) > 1:
                    params["zoneb_layers"] = zones_data[1]["layers"]
                    params["zoneb_material"] = zones_data[1]["material"]
                params["material_dep"] = zones_data[0]["material"]
        else:
            params.pop("zones", None)
            params["zonea_layers"] = params["n_layers"]
            params["zonea_material"] = params["material_dep"]
            params["zoneb_layers"] = 0
            params["zoneb_material"] = params["material_dep"]
        # ---------------------------------

        
        # Mechanical Boundary Condition scheme
        bc_str = getattr(self, "mech_bc_var", None)
        if bc_str:
            val = bc_str.get()
            if "Cantilever" in val:
                params["mechanical_bc"] = "cantilever"
            elif "Clamped" in val:
                params["mechanical_bc"] = "fixture_clamped"
            else:
                params["mechanical_bc"] = "isostatic"

        # Heat Input & Power Calculations
        I = float(params.get("I_weld", 180.0))
        U = float(params.get("U_weld", 20.0))
        eta = float(params.get("eta_arc", 0.8))
        v_m_s = float(params.get("v_scan", 0.008))
        v_mm_s = v_m_s * 1000.0
        p_gross = U * I
        p_net = eta * p_gross
        h_net_j_mm = p_net / v_mm_s if v_mm_s > 0 else 0.0
        h_gross_j_mm = p_gross / v_mm_s if v_mm_s > 0 else 0.0

        params["power_gross_W"] = round(p_gross, 2)
        params["power_net_W"] = round(p_net, 2)
        params["heat_input_net_J_per_mm"] = round(h_net_j_mm, 2)
        params["heat_input_gross_J_per_mm"] = round(h_gross_j_mm, 2)
        return params

    def _log(self, msg: str):
        if not hasattr(self, "console"):
            return
        self.console.config(state=tk.NORMAL)
        text = msg if msg.endswith("\n") else msg + "\n"
        s = text.strip()

        # Insert clean line spacing before major phase/layer transitions so lines don't clutter
        is_major_section = (
            s.startswith("--- Layer") or
            "Inter-pass Cooling" in s or
            "FINAL COOLING" in s or
            "MECHANICAL STAGE" in s or
            "[OK] SIMULATION COMPLETED" in s or
            "STARTING WAAM MULTI-PHYSICS SIMULATION" in s
        )
        if is_major_section:
            try:
                prev = self.console.get("end-2c", "end-1c")
                if prev and prev != "\n":
                    self.console.insert(tk.END, "\n")
            except Exception:
                pass

        # Classify syntax highlighting tag
        tag = None
        if "[ERROR]" in text or "[Error]" in text or "Execution failure" in text or "exited with code" in text:
            tag = "error"
        elif "[WARN]" in text or "[WARNING]" in text or "[Warning]" in text:
            tag = "warn"
        elif "[OK]" in text or "SIMULATION COMPLETED" in text or "Normal End" in text or "✓ Status:" in text:
            tag = "ok"
        elif s.startswith("--- Layer") or ("Layer " in s and "Deposition" in s):
            tag = "layer"
        elif "Inter-pass" in s or "FINAL COOLING" in s or "MECHANICAL STAGE" in s:
            tag = "phase"
        elif s.startswith("ws=") or s.startswith("Step:"):
            tag = "step"
        elif any(k in text for k in ("[SUMMARY]", "[PLOTS]", "[PARAVIEW]", "[REPORT]", "[LOG]", "[INFO]", "[SIM]")):
            tag = "info"
        elif s.startswith("•") or s.startswith("  •"):
            tag = "bullet"
        elif s.startswith("=") and len(s) >= 15:
            tag = "banner"
        elif any(s.startswith(p) for p in ("Timestamp:", "Script:", "Case Dir:", "Run Dir:", "FreeFem++:")):
            tag = "dim"

        if tag:
            self.console.insert(tk.END, text, tag)
        else:
            self.console.insert(tk.END, text)

        # Respect auto-scroll toggle
        if getattr(self, "console_autoscroll_var", None) is None or self.console_autoscroll_var.get():
            self.console.see(tk.END)

        self.console.config(state=tk.DISABLED)

    def _get_base_output_dir(self) -> str:
        """Get the root waam_output directory configured in the UI (always inside WAAM-Framework/waam_output)."""
        _framework_dir = Path(__file__).resolve().parent.parent
        default_out = (_framework_dir / "waam_output").resolve()
        base = self._param_vars.get("output_dir", (tk.StringVar(value=str(default_out)),))[0].get()
        if not base or not base.strip() or base in ("./waam_output", "./waam_output/", "waam_output", "waam_output/"):
            base = str(default_out)
        p = Path(base)
        if not p.is_absolute():
            p = (_framework_dir / base).resolve()
        else:
            p = p.resolve()
        # If pointing to outer workspace /Users/.../Desktop/WAAM/waam_output, redirect to WAAM-Framework/waam_output
        if str(p) == str(_framework_dir.parent / "waam_output"):
            p = default_out
        if re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", p.name):
            base = str(p.parent)
        else:
            base = str(p)
        return os.path.abspath(base)

    def _get_or_create_run_dir(self, is_new_run: bool = False) -> str:
        """
        Manage timestamped run directory inside waam_output/.
        - When EDP script is generated: creates or reuses active timestamped folder.
        - Until a new run simulation is clicked, does not create and redirect to a new folder.
        - When a new run simulation is clicked (and previous run was already executed), creates a new timestamped folder.
        """
        import datetime
        base_dir = self._get_base_output_dir()
        os.makedirs(base_dir, exist_ok=True)

        if self.active_run_dir is None or (is_new_run and self.active_run_executed):
            now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = os.path.join(base_dir, now_str)
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, "FreeFEM", "visual_outputs"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "openfoam"), exist_ok=True)
            self.active_run_dir = os.path.abspath(run_dir)
            self.active_run_executed = False
            return self.active_run_dir

        if not os.path.exists(self.active_run_dir):
            os.makedirs(self.active_run_dir, exist_ok=True)
        fem_vis = os.path.join(self.active_run_dir, "FreeFEM", "visual_outputs")
        if not os.path.exists(fem_vis):
            os.makedirs(fem_vis, exist_ok=True)

        return self.active_run_dir

    @staticmethod
    def _fem_subdir(run_dir: str) -> str:
        """FreeFEM side of a case folder: <run_dir>/FreeFEM/."""
        return os.path.join(run_dir, "FreeFEM")

    def _fem_vis_dir(self, run_dir: str) -> str:
        """Visual outputs of the FreeFEM side (legacy top-level fallback handled by callers)."""
        return os.path.join(self._fem_subdir(run_dir), "visual_outputs")

    def _get_active_or_latest_run_dir(self) -> str:
        """Return active run directory, or scan base output directory for latest timestamped folder."""
        if self.active_run_dir and os.path.exists(self.active_run_dir):
            return self.active_run_dir
        base_dir = self._get_base_output_dir()
        if os.path.exists(base_dir):
            subdirs = [
                os.path.join(base_dir, d) for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", d)
            ]
            if subdirs:
                subdirs.sort(reverse=True)
                self.active_run_dir = os.path.abspath(subdirs[0])
                return self.active_run_dir
        return base_dir

    def _generate_script(self, is_new_run: bool = False):
        try:
            run_dir = self._get_or_create_run_dir(is_new_run=is_new_run)
            fem_dir = self._fem_subdir(run_dir)
            vis_dir = os.path.join(fem_dir, "visual_outputs")
            os.makedirs(vis_dir, exist_ok=True)

            params = self._collect_params()
            params["output_dir"] = fem_dir.rstrip('/') + '/'

            # If DEFAULT preset is selected, check whether custom CAD/mesh is loaded
            preset_name = self.preset_var.get() if hasattr(self, "preset_var") else ""
            has_custom_cad = bool(
                params.get("stl_file") or
                params.get("mesh_file") or
                (hasattr(self, "mesh_source_var") and self.mesh_source_var.get() in ("cad_file", "gcode_file")) or
                (hasattr(self, "loaded_stl_path") and self.loaded_stl_path)
            )

            # Always use EDPBuilder to generate multi-physics script dynamically from params

            if has_custom_cad:
                stl_cand = params.get("stl_file") or getattr(self, "loaded_stl_path", "")
                existing_mesh = params.get("mesh_file", "")
                
                if existing_mesh and os.path.exists(existing_mesh) and existing_mesh.lower().endswith(('.mesh', '.mesh3', '.msh')):
                    # We already have a valid mesh file (e.g. from the import thread)
                    params["mesh_source"] = "cad_file"
                    # Just copy it to the fem_dir for self-contained runs
                    mesh_out = os.path.join(fem_dir, os.path.basename(existing_mesh))
                    if os.path.abspath(existing_mesh) != os.path.abspath(mesh_out):
                        try:
                            import shutil
                            shutil.copy2(existing_mesh, mesh_out)
                            params["mesh_file"] = mesh_out
                        except Exception as e:
                            self._log(f"[SIM] Warning: could not copy mesh file: {e}\n")
                elif stl_cand and os.path.exists(stl_cand) and Path(stl_cand).suffix.lower() in ('.stl', '.step', '.stp', '.iges', '.igs', '.ply', '.obj'):
                    mesh_out = os.path.join(fem_dir, Path(stl_cand).stem + ".mesh")
                    if not os.path.exists(mesh_out) or os.path.getmtime(mesh_out) < os.path.getmtime(stl_cand):
                        self._log(f"\n[CAD-SIM] Automatically meshing {os.path.basename(stl_cand)} to FreeFEM volume mesh...\n")
                        from mesh.stl_pipeline import stl_to_freefem_mesh
                        target_nodes = 2000
                        try:
                            target_nodes = int(self.mesh_budget_var.get())
                        except Exception:
                            pass
                        res = stl_to_freefem_mesh(stl_cand, mesh_out, target_nodes=target_nodes)
                        self._log(f"[CAD-SIM] Volume mesh ready: {res['n_nodes']} nodes, {res['n_tets']} tets -> {mesh_out}\n")
                    params["mesh_file"] = mesh_out
                    params["mesh_source"] = "cad_file"
                    params["stl_file"] = stl_cand
                    self._log(f"[CAD-SIM] Applying authentic DEFAULT.edp multi-physics conditions to CAD model: {os.path.basename(stl_cand)}\n")



            # If standard box wall or no custom CAD/gcode specified, auto-update wall toolpath files
            if params.get("mesh_source", "box_wall") == "box_wall" or not params.get("gcode_file"):
                self._auto_update_wall_gcode(run_dir=fem_dir)
                auto_gc = os.path.join(fem_dir, "toolpath.gcode")
                if os.path.exists(auto_gc):
                    params["gcode_file"] = auto_gc

            builder = EDPBuilder(params)
            for w in getattr(builder, "warnings", []):
                self._log(f"[ADVISORY] [INFO] {w}\n")

            # If custom G-code or CAD/STL was selected, copy it to the case folder
            # (shared inputs at top level: both FreeFEM and OpenFOAM sides use them)
            import shutil
            for key in ["gcode_file", "stl_file", "mesh_file"]:
                fpath = params.get(key)
                if fpath and os.path.exists(fpath) and os.path.isfile(fpath):
                    dest = os.path.join(run_dir, os.path.basename(fpath))
                    if os.path.abspath(fpath) != os.path.abspath(dest):
                        try:
                            shutil.copy2(fpath, dest)
                            params[key] = dest
                            self._log(f"[FILE SYNC] Copied {os.path.basename(fpath)} to run folder: {dest}\n")
                        except Exception as fe:
                            self._log(f"[FILE SYNC] Warning: {fe}\n")

            script_path = os.path.join(fem_dir, "waam_simulation.edp")
            builder.save(script_path)

            h_net = params.get("heat_input_net_J_per_mm", 0.0)
            p_net = params.get("power_net_W", 0.0)
            h_gr = params.get("heat_input_gross_J_per_mm", 0.0)
            self._log(f"\n[OUTPUT DIRECTORY] Directed to: {fem_dir}\n")
            self._log(f"[VISUAL OUTPUTS]   Directed to: {vis_dir}\n")
            self._log(f"[PROCESS] Calculated Heat Input: Net = {h_net:.2f} J/mm ({h_net/1000.0:.3f} kJ/mm) | P_net = {p_net/1000.0:.2f} kW | Gross = {h_gr:.2f} J/mm\n")
            self._log(f"[CODEGEN] [OK] FreeFEM++ script generated: {script_path}\n")
            self.status_var.set("● Status: Script Generated")

            params_path = os.path.join(fem_dir, "parameters.json")
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)
            shutil.copy2(params_path, os.path.join(run_dir, "parameters.json"))
            self._log(f"[CODEGEN] Parameters metadata saved: {params_path}\n")
            return script_path
        except Exception as e:
            self._log(f"\n[CODEGEN] [ERROR] Error generating script: {e}\n")
            messagebox.showerror("Script Generation Error", str(e))
            return None

    def _run_simulation(self):
        # Until a new run simulation is not clicked do not create and redirect to new folder.
        # If the active run directory has already completed a run, clicking RUN SIMULATION
        # creates and redirects to a fresh datetime run directory.
        is_new = bool(self.active_run_executed)
        script_path = self._generate_script(is_new_run=is_new)
        if not script_path:
            return

        self.active_run_executed = True
        self._log(f"\n{'='*55}\nSTARTING WAAM MULTI-PHYSICS SIMULATION...\n{'='*55}\n")
        self.running = True
        self.status_var.set("▶ Status: Solving...")
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        if hasattr(self, "console_status_dot"):
            self.console_status_dot.config(text="● STREAMING LIVE", fg="#22c55e")

        def run_thread():
            rc = -1
            try:
                # Portable FreeFem++ discovery: PATH/shutil.which first, then macOS bundle
                ff_path = shutil.which("FreeFem++")
                if not ff_path:
                    _bundle = "/Applications/FreeFem++.app/Contents/ff-4.15/bin/FreeFem++"
                    ff_path = _bundle if os.path.exists(_bundle) else "FreeFem++"

                abs_script = os.path.abspath(script_path)
                abs_cwd = os.path.abspath(os.path.dirname(script_path) or ".")

                # MULTI-CORE CPU: OMP/MKL/VECLIB threads (caps at 12 to leave
                # headroom for OS + GUI on Apple Silicon).
                env = os.environ.copy()
                n_cores = str(min(os.cpu_count() or 4, 12))
                env["OMP_NUM_THREADS"] = n_cores
                env["OPENBLAS_NUM_THREADS"] = n_cores
                env["MKL_NUM_THREADS"] = n_cores
                env["VECLIB_MAXIMUM_THREADS"] = n_cores  # Apple Accelerate Framework on macOS

                import re
                # Case layout: $case/{FreeFEM,openfoam}. The script lives in the
                # FreeFEM side, so FreeFEM outputs stay there; top-level HTML
                # copies are published after postprocessing.
                if os.path.basename(os.path.normpath(abs_cwd)) == "FreeFEM":
                    fem_dir = abs_cwd
                    case_dir = os.path.dirname(abs_cwd)
                else:  # legacy flat case: FreeFEM side is created underneath
                    case_dir = self.active_run_dir or abs_cwd
                    fem_dir = os.path.join(case_dir, "FreeFEM")
                os.makedirs(fem_dir, exist_ok=True)
                target_run_dir = fem_dir

                # LIVE OUTPUT FIX: FreeFem++ (compiled C++) buffers stdout when
                # it detects a pipe instead of a terminal (isatty() check).
                # Wrapping in `script -q /dev/null` creates a pseudo-terminal
                # that forces line-by-line flushing, matching the behaviour you
                # see when running directly in a terminal.
                _use_script = shutil.which("script") is not None  # macOS/Linux always has this
                if _use_script:
                    cmd = ["script", "-q", "/dev/null", ff_path, abs_script, os.path.abspath(target_run_dir) + "/"]
                else:
                    cmd = [ff_path, abs_script, os.path.abspath(target_run_dir) + "/"]

                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=abs_cwd,
                    env=env,
                    bufsize=1
                )
                log_file_path = os.path.join(fem_dir, "sim.log")
                with open(log_file_path, "w", encoding="utf-8") as log_f:
                    log_f.write("================================================================================\n")
                    log_f.write("WAAM MULTI-PHYSICS SIMULATION TERMINAL LOG\n")
                    log_f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_f.write(f"Script:    {abs_script}\n")
                    log_f.write(f"Case Dir:  {case_dir}\n")
                    log_f.write(f"Run Dir:   {target_run_dir}\n")
                    log_f.write(f"FreeFem++: {ff_path}\n")
                    log_f.write("================================================================================\n\n")
                    log_f.flush()

                    for line in self.process.stdout:
                        if not self.running:
                            break
                        # Strip CR/control chars injected by `script` pseudo-terminal (preserve newlines/tabs)
                        line = line.replace("\r", "")
                        while line and ord(line[0]) < 32 and line[0] not in "\n\t":
                            line = line[1:]
                        log_f.write(line)
                        log_f.flush()

                        # Stream complete authentic terminal output live to GUI console
                        self.root.after(0, self._log, line)

                        # Update stage & progress tracker
                        m_lay = re.search(r"--- Layer\s+(\d+)/(\d+)", line)
                        if m_lay:
                            cl, tl = int(m_lay.group(1)), int(m_lay.group(2))
                            p = int((cl / tl) * 80.0)
                            self.root.after(0, lambda c=cl, t=tl, val=p: (
                                self.progress_bar.configure(value=val),
                                self.stage_label.configure(text=f"Current Phase: Layer {c} of {t} (Deposition )"),
                            ))

                        m_ws = re.search(r"ws=(\d+).*?Tmax=([\d\.]+)", line)
                        if m_ws:
                            w_step = m_ws.group(1)
                            t_val = m_ws.group(2)
                            self.root.after(0, lambda w=w_step, tv=t_val: (
                                self.step_detail_label.configure(text=f"Step: {w} | Peak Temp: {tv} °C | Status: Active Arc Melting")
                            ))

                        if "Inter-pass" in line:
                            self.root.after(0, lambda: self.stage_label.configure(
                                text="Current Phase: Inter-Pass Cooling Dwell "
                            ))

                        if "FINAL COOLING" in line:
                            self.root.after(0, lambda: (
                                self.progress_bar.configure(value=85),
                                self.stage_label.configure(text="Current Phase: Final Part Cooling to Room Temp ")
                            ))

                        if "MECHANICAL STAGE" in line:
                            self.root.after(0, lambda: (
                                self.progress_bar.configure(value=95),
                                self.stage_label.configure(text="Current Phase: Solving Residual Stress & Distortion")
                            ))

                        if "DONE" in line or "Normal End" in line:
                            self.root.after(0, lambda: (
                                self.progress_bar.configure(value=100),
                                self.stage_label.configure(text="Current Phase: Simulation Complete! [OK]"),
                                self.step_detail_label.configure(text="Status: Finished. Ready for ParaView visualization.")
                            ))

                    self.process.wait()
                    rc = self.process.returncode

                    log_f.write("\n================================================================================\n")
                    log_f.write(f"EXECUTION FINISHED - RETURN CODE: {rc}\n")
                    log_f.write(f"Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_f.write("================================================================================\n")
                    log_f.flush()

                if rc == 0:
                    self.root.after(0, self.status_var.set, "✓ Status: Completed")
                    if hasattr(self, "console_status_dot"):
                        self.root.after(0, lambda: self.console_status_dot.config(text="● COMPLETED", fg="#38bdf8"))
                    try:
                        from core.postprocess.output_manager import (
                            generate_run_summary,
                            generate_paraview_loader_script,
                        )
                        from postprocess.plot_generator import generate_all_plots
                        p_final = self._collect_params()
                        p_final["output_dir"] = target_run_dir.rstrip('/') + '/'
                        plots = generate_all_plots(target_run_dir)
                        pv_scr = generate_paraview_loader_script(target_run_dir)
                        summary = generate_run_summary(target_run_dir, p_final, return_code=rc)

                        summary_file = os.path.join(target_run_dir, 'run_summary.json')
                        plots_dir = os.path.join(target_run_dir, 'plots')

                        # Clean, spacious delivery presentation with line gaps
                        sep_line = "=" * 76
                        banner_msg = (
                            f"\n\n{sep_line}\n"
                            f"  [OK] SIMULATION COMPLETED SUCCESSFULLY\n"
                            f"{sep_line}\n\n"
                            f"  • [SUMMARY]   Executive run summary saved:\n"
                            f"                {summary_file}\n\n"
                            f"  • [PLOTS]     Generated {len(plots)} publication-quality 2D analysis plot(s):\n"
                            f"                {plots_dir}\n\n"
                            f"  • [PARAVIEW]  3D visualization loader script created:\n"
                            f"                {pv_scr}\n\n"
                        )
                        self.root.after(0, self._log, banner_msg)

                        try:
                            from core.postprocess.output_manager import generate_enhanced_html_report
                            report_html = generate_enhanced_html_report(target_run_dir, p_final, summary, plots)
                            if report_html and os.path.exists(report_html):
                                top_rep = os.path.join(case_dir, "report.html")
                                if top_rep != report_html:
                                    import shutil as _sh
                                    _sh.copy2(report_html, top_rep)
                                self.root.after(0, self._log, f"  • [REPORT]    Comprehensive engineering HTML report:\n                {report_html}\n\n")
                            import shutil as _sh
                            top_log = os.path.join(case_dir, "sim.log")
                            if os.path.exists(log_file_path) and top_log != log_file_path:
                                _sh.copy2(log_file_path, top_log)
                                self.root.after(0, self._log, f"  • [LOG]       Complete execution log published:\n                {top_log}\n\n")
                        except Exception as rep_err:
                            self.root.after(0, self._log, f"  • [REPORT]    Publishing note: {rep_err}\n\n")

                        self.root.after(0, self._log, f"{sep_line}\n\n")

                        self.root.after(500, self._log_reload)
                    except Exception as pe:
                        self.root.after(0, self._log, f"[POSTPROCESS] [WARNING] Post-processing note: {pe}\n")
                else:
                    self.root.after(0, self._log, f"\n[ERROR] FreeFEM exited with code {rc}\n")
                    self.root.after(0, self.status_var.set, f"✕ Status: Error (Exit {rc})")
                    if hasattr(self, "console_status_dot"):
                        self.root.after(0, lambda: self.console_status_dot.config(text="● ERROR", fg="#ef4444"))
            except Exception as e:
                self.root.after(0, self._log, f"\n[ERROR] Execution failure: {e}\n")
                self.root.after(0, self.status_var.set, "✕ Status: Execution Error")
                if hasattr(self, "console_status_dot"):
                    self.root.after(0, lambda: self.console_status_dot.config(text="● ERROR", fg="#ef4444"))
            finally:
                self.running = False
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
                if hasattr(self, "console_status_dot"):
                    self.root.after(0, lambda: self.console_status_dot.config(
                        text="● COMPLETED" if rc == 0 else "● STOPPED",
                        fg="#38bdf8" if rc == 0 else "#f59e0b"
                    ))

        threading.Thread(target=run_thread, daemon=True).start()

    def _confirm_quit(self):
        """Quit with confirm when a simulation is running."""
        if getattr(self, "running", False):
            if not messagebox.askyesno(
                "Quit",
                "A simulation is running. Quit anyway?\nThe solver process will be stopped.",
            ):
                return
            try:
                self._stop_simulation()
            except Exception:
                pass
        self.root.quit()

    def _stop_simulation(self):
        if self.running:
            if not messagebox.askyesno(
                "Stop Simulation",
                "A simulation is running. Stop it now?\nPartial results will be kept.",
            ):
                return
        self.running = False
        if hasattr(self, "console_status_dot"):
            self.console_status_dot.config(text="● STOPPED", fg="#f59e0b")
        if self.process:
            # terminate() only signals the direct child, which is the `script`
            # pty wrapper -- the FreeFEM process it spawned keeps running and
            # keeps burning a core. Signal the whole group.
            proc = self.process
            for sig, kw in ((None, {}), ("group", {})):
                try:
                    if sig == "group" and hasattr(os, "killpg"):
                        os.killpg(os.getpgid(proc.pid), 9)
                    else:
                        proc.terminate()
                    break
                except Exception:
                    continue
            try:
                proc.wait(timeout=8)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._log("\nProcess terminated by user.\n")
        self.status_var.set("■ Status: Stopped")
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _open_output(self):
        outdir = self._get_active_or_latest_run_dir()
        os.makedirs(outdir, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", outdir])
        elif sys.platform == "linux":
            subprocess.Popen(["xdg-open", outdir])
        else:
            subprocess.Popen(["explorer", outdir])

    def _open_report(self):
        """Open the auto-generated HTML simulation summary report in the browser."""
        import glob
        outdir = self._get_active_or_latest_run_dir()
        # Prioritize interactive waam_report.html, then standard report.html
        search_order = [
            os.path.abspath(os.path.join(outdir, "waam_report.html")),
            os.path.abspath(os.path.join(outdir, "FreeFEM", "waam_report.html")),
            os.path.abspath(os.path.join(outdir, "report.html")),
            os.path.abspath(os.path.join(outdir, "FreeFEM", "report.html")),
        ]
        report_path = next((c for c in search_order if os.path.exists(c)), search_order[0])

        # 1. If report exists in active/latest run dir, open it
        if os.path.exists(report_path):
            import webbrowser
            webbrowser.open(f"file://{report_path}")
            self._log(f"[REPORT] Opened simulation report: {report_path}\n")
            return

        # Open plots directory with native viewer/finder
        plots_dir = os.path.join(outdir, "plots")
        if not os.path.exists(plots_dir):
            plots_dir = os.path.join(outdir, "FreeFEM", "plots")
        if not os.path.exists(plots_dir):
            all_plots = glob.glob(os.path.join(self._get_base_output_dir(), "*", "plots"))
            all_plots += glob.glob(os.path.join(self._get_base_output_dir(), "*", "FreeFEM", "plots"))
            if all_plots:
                plots_dir = max(all_plots, key=os.path.getmtime)
        if os.path.exists(plots_dir):
            if sys.platform == "darwin":
                subprocess.run(["open", plots_dir])
            elif sys.platform == "win32":
                os.startfile(plots_dir)
            else:
                subprocess.run(["xdg-open", plots_dir])
            self._log(f"[PLOTS] Opened plots directory: {plots_dir}\n")
            return

        messagebox.showinfo(
            "Plots Not Found",
            "No plots found yet. Run a simulation first to produce results."
        )

    def _generate_html_report(self, params: dict):
        """Generate a self-contained comprehensive HTML simulation summary report."""
        base = params.get("output_dir", "./waam_output")
        # Route into the FreeFEM side unless the caller already points there.
        outdir = base if os.path.basename(os.path.normpath(base)) == "FreeFEM" else os.path.join(base, "FreeFEM")
        os.makedirs(outdir, exist_ok=True)
        try:
            from core.postprocess.output_manager import generate_enhanced_html_report, generate_run_summary
            summary = generate_run_summary(outdir, params)
            report_path = generate_enhanced_html_report(outdir, params, summary)
            import shutil as _sh
            _sh.copy2(report_path, os.path.join(os.path.dirname(outdir.rstrip('/')), "report.html"))
            return report_path
        except Exception as e:
            self._log(f"[REPORT] Generation note: {e}\n")
            report_path = os.path.join(outdir, "report.html")
            with open(report_path, "w") as f:
                f.write("<!DOCTYPE html><html><body><h1>WAAM Report</h1></body></html>")
            return report_path

    def _save_project(self):
        """Save current parameters to a .waam JSON project file."""
        path = filedialog.asksaveasfilename(
            defaultextension=".waam",
            filetypes=[("WAAM Project", "*.waam"), ("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Save WAAM Project"
        )
        if not path:
            return
        params = self._collect_params()
        import datetime
        params["_project_meta"] = {
            "saved": datetime.datetime.now().isoformat(),
            "framework_version": "2.0",
            "description": "WAAM-Framework project file"
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
        self._log(f"[PROJECT] Saved project to: {path}\n")
        messagebox.showinfo("Project Saved", f"Project saved:\n{path}")

    def _load_project(self):
        """Load a .waam JSON project file and restore all parameters."""
        path = filedialog.askopenfilename(
            filetypes=[("WAAM Project", "*.waam"), ("JSON Files", "*.json"), ("All Files", "*.*")],
            title="Load WAAM Project"
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not read project file:\n{e}")
            return

        # Restore numeric params
        for key, (var, type_) in self._param_vars.items():
            if key not in loaded:
                continue
            raw = loaded[key]
            # Convert SI meters back to mm for display keys
            if key in MM_DIMENSION_KEYS and isinstance(raw, (int, float)):
                raw = round(raw * 1000.0, 4)
            elif key in MM_SPEED_KEYS and isinstance(raw, (int, float)):
                raw = round(raw * 1000.0, 4)
            var.set(str(raw))

        # Restore combos and radio buttons
        if "heat_source" in loaded and hasattr(self, "source_var"):
            self.source_var.set(loaded["heat_source"])
            self._update_source_params()
        if "material_base" in loaded and hasattr(self, "material_base_var"):
            self.material_base_var.set(loaded["material_base"])
        if "material_dep" in loaded and hasattr(self, "material_dep_var"):
            self.material_dep_var.set(loaded["material_dep"])
            self._on_multi_material_change()

        self._update_mesh_calc()
        self._update_heat_calc()
        self._log(f"[PROJECT] Loaded project from: {path}\n")
        messagebox.showinfo("Project Loaded", f"Project loaded:\n{path}")

    def _import_stl(self):
        """Import an STL file, update geometry dimensions, and mesh it using Gmsh for FreeFEM."""
        stl_path = filedialog.askopenfilename(
            filetypes=[("STL Files", "*.stl"), ("CAD Files", "*.step *.stp *.iges *.igs"), ("All Files", "*.*")],
            title="Select STL / CAD File for Import & Meshing",
            initialdir=str(Path.home() / "Desktop")
        )
        if not stl_path:
            return
        self._handle_cad_stl_selected(stl_path, auto_mesh=True)


    def _open_paraview(self):
        """Open simulation results in ParaView, targeting the FreeFEM visual_outputs."""
        run_dir = self._get_active_or_latest_run_dir()
        vis_candidates = [
            os.path.join(run_dir, "FreeFEM", "visual_outputs"),
            os.path.join(run_dir, "visual_outputs"),  # legacy flat layout
        ]
        vis_dir = next((v for v in vis_candidates if os.path.isdir(v)), vis_candidates[0])

        micro_vtu = os.path.abspath(os.path.join(vis_dir, "waam_micro.vtu"))
        mech_vtu = os.path.abspath(os.path.join(vis_dir, "waam_mech.vtu"))
        pvd_file = os.path.abspath(os.path.join(vis_dir, "waam.pvd"))

        # Fallback to run_dir root if someone has older legacy files
        if not os.path.exists(micro_vtu) and os.path.exists(os.path.join(run_dir, "waam_micro.vtu")):
            micro_vtu = os.path.abspath(os.path.join(run_dir, "waam_micro.vtu"))
        if not os.path.exists(mech_vtu) and os.path.exists(os.path.join(run_dir, "waam_mech.vtu")):
            mech_vtu = os.path.abspath(os.path.join(run_dir, "waam_mech.vtu"))
        if not os.path.exists(pvd_file) and os.path.exists(os.path.join(run_dir, "waam.pvd")):
            pvd_file = os.path.abspath(os.path.join(run_dir, "waam.pvd"))

        targets = []
        if os.path.exists(micro_vtu):
            targets.append(micro_vtu)
        if os.path.exists(mech_vtu):
            targets.append(mech_vtu)
        if os.path.exists(pvd_file):
            targets.append(pvd_file)

        if not targets:
            sub_targets = glob.glob(os.path.join(vis_dir, "*.pvd")) + glob.glob(os.path.join(vis_dir, "*.vtu"))
            for legacy in (run_dir, os.path.join(run_dir, "FreeFEM")):
                if not sub_targets:
                    sub_targets = glob.glob(os.path.join(legacy, "*.pvd")) + glob.glob(os.path.join(legacy, "*.vtu"))
            targets = [os.path.abspath(f) for f in sorted(list(set(sub_targets)))]

        if not targets:
            self._log(f"[PARAVIEW] [WARNING] No VTU/PVD visual datasets found in:\n  {vis_dir}\n"
                      f"Execute 'RUN SIMULATION' first to generate visual outputs.\n")
            messagebox.showinfo(
                "No Visual Outputs Found",
                f"No 3D visual outputs (.vtu or .pvd) found in:\n{vis_dir}\n\n"
                f"Click 'RUN SIMULATION' to solve the case and generate ParaView datasets."
            )
            return

        # Locate ParaView binary (portable: PATH/env first, then well-known spots)
        candidates = [
            shutil.which("paraview"),
            os.environ.get("PARAVIEW_BIN"),
            os.path.expanduser("~/bin/paraview"),
            "/Applications/ParaView-5.13.3.app/Contents/MacOS/paraview",
            "/opt/homebrew/bin/paraview",
            "/usr/local/bin/paraview",
        ] + glob.glob("/Applications/ParaView*.app/Contents/MacOS/paraview")

        pv_bin = None
        for c in candidates:
            if c and os.path.exists(c):
                pv_bin = c
                break

        opened = False
        if pv_bin:
            try:
                subprocess.Popen([pv_bin] + targets)
                self._log(f"[PARAVIEW] [OK] Launching {pv_bin}\n")
                for t in targets:
                    self._log(f"  -> {os.path.basename(t)} ({os.path.dirname(t)})\n")
                opened = True
            except Exception as e:
                self._log(f"[PARAVIEW] [WARNING] Binary launch error: {e}\n")

        if not opened and sys.platform == "darwin":
            pv_apps = ["/Applications/ParaView-5.13.3.app", "/Applications/ParaView.app"] + glob.glob("/Applications/ParaView*.app")
            found = [p for p in pv_apps if os.path.exists(p)]
            if found:
                try:
                    subprocess.Popen(["open", "-a", found[0], "--args"] + targets)
                    self._log(f"[PARAVIEW] [OK] Opening via open -a {found[0]} ({len(targets)} files)\n")
                    opened = True
                except Exception as e:
                    self._log(f"[PARAVIEW] [ERROR] open -a failed: {e}\n")

        if not opened:
            self._open_output()
            self._log(f"[PARAVIEW] [INFO] ParaView binary not found. Datasets are in: {vis_dir}\n")
            messagebox.showinfo(
                "ParaView Not Found",
                f"ParaView application was not found on your system.\n\n"
                f"Your visual output datasets are saved in:\n{vis_dir}\n\n"
                f"Files ready for inspection:\n" + "\n".join(f" - {os.path.basename(t)}" for t in targets) + "\n\n"
                f"To view in 3D, download ParaView from: https://www.paraview.org"
            )

    def _openfoam_dialog(self):
        """Legacy toolbar button handler -- redirects to the OpenFOAM tab generate."""
        self._openfoam_generate_case()

    # The multi-material study directory was created with a transposed
    # letter ("materaial"). The on-disk name is kept so existing checkouts keep
    # working; every reference goes through this constant, and the correctly
    # spelled name is accepted as a fallback.
    CLAUDE_DIR_NAME = "multi_materaial_claude"
    CLAUDE_DIR_NAME_FIXED = "multi_material_claude"

    def _claude_dir(self):
        _here = Path(__file__).resolve().parent.parent
        legacy = _here / self.CLAUDE_DIR_NAME
        if legacy.exists():
            return legacy
        return _here / self.CLAUDE_DIR_NAME_FIXED

    def _run_claude_simulation(self, preset=1, custom_stdin=None):
        """Run the multi-material FreeFEM++ simulation with the specified preset or custom configuration."""
        if self.running:
            messagebox.showwarning("Simulation Running", "A simulation is already in progress. Please stop it first.")
            return

        claude_dir = self._claude_dir()
        main_edp = claude_dir / "main.edp"
        if not main_edp.exists():
            messagebox.showerror("File Not Found", f"Cannot find {main_edp}")
            return

        preset_labels = {
            1: "Default Benchmark (10 Layers, 20 mm Wall)",
            2: "Custom Multi-Material Zone Configuration",
            3: "100 mm Tall Wall Build (50 Layers, 100 mm Wall)"
        }
        lbl = preset_labels.get(preset, f"Preset {preset}")
        self._log(f"\n{'='*65}\nSTARTING MULTI-MATERIAL SIMULATION: {lbl}...\n{'='*65}\n")
        self.running = True
        self.status_var.set("▶ Status: Multi-Material Solving...")
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        def thread_fn():
            try:
                ff_path = shutil.which("FreeFem++")
                if not ff_path:
                    _bundle = "/Applications/FreeFem++.app/Contents/ff-4.15/bin/FreeFem++"
                    ff_path = _bundle if os.path.exists(_bundle) else "FreeFem++"

                env = os.environ.copy()
                n_cores = str(min(os.cpu_count() or 4, 12))
                env["OMP_NUM_THREADS"] = n_cores
                env["OPENBLAS_NUM_THREADS"] = n_cores
                env["MKL_NUM_THREADS"] = n_cores
                env["VECLIB_MAXIMUM_THREADS"] = n_cores

                run_dir = self._get_or_create_run_dir(is_new_run=True)
                cmd = [ff_path, "main.edp", os.path.abspath(run_dir) + "/"]
                self.process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(claude_dir),
                    env=env,
                    bufsize=1
                )

                # Send preset choice or custom configuration sequence to stdin,
                # then CLOSE it. A main.edp that reads past what was provided
                # used to block forever on the open pipe: the run never
                # finished, Stop could not join this thread, and the orphaned
                # FreeFEM kept the core.
                if preset == 2 and custom_stdin:
                    self.process.stdin.write(custom_stdin)
                else:
                    self.process.stdin.write(f"{preset}\n")
                self.process.stdin.flush()
                try:
                    self.process.stdin.close()
                except Exception:
                    pass

                for line in iter(self.process.stdout.readline, ''):
                    if not self.running:
                        break
                    line_clean = line.strip()
                    if line_clean:
                        self.root.after(0, self._log, line_clean)
                    if "Layer " in line and "z=[" in line:
                        msg = line.strip()
                        self.root.after(0, lambda m=msg: self.stage_label.config(text=m))
                    if "Max Released Springback:" in line:
                        msg = line.strip()
                        self.root.after(0, lambda m=msg: self.step_detail_label.config(text=m))

                self.process.stdout.close()
                self.process.wait()

                try:
                    from postprocess.plot_generator import generate_all_plots
                    plots = generate_all_plots(run_dir)
                    self.root.after(0, self._log, f"[PLOTS] Generated {len(plots)} publication-quality plots in: {os.path.join(run_dir, 'plots')}\n")
                except Exception as ex:
                    pass
            except Exception as ex:
                self.root.after(0, self._log, f"[ERROR] Multi-Material solver execution failed: {ex}")
            finally:
                self.running = False
                self.root.after(0, lambda: self.status_var.set("● Status: Ready"))
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
                self.root.after(0, self._log, f"[OK] Multi-Material Simulation Complete.")

        t = threading.Thread(target=thread_fn, daemon=True)
        t.start()

    def _run_claude_dialog(self):
        """Open detailed custom multi-material configuration dialog with arbitrary zone and layer allocations."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Multi-Material Custom Configuration")
        dlg.geometry("580x680")
        dlg.transient(self.root)
        dlg.grab_set()

        mat_names = ["SS308L", "SS316L", "AL1XXX", "ER70S"]
        mat_map = {"SS308L": 0, "SS316L": 1, "AL1XXX": 2, "ER70S": 3}

        # Header
        top_frame = tk.Frame(dlg, padx=16, pady=10)
        top_frame.pack(fill=tk.X)
        tk.Label(top_frame, text="Custom Multi-Material Configuration", font=FONT_DLG_TITLE, fg="#111827").pack(anchor=tk.W)
        tk.Label(top_frame, text="Configure number of materials, total layers, and zone-by-zone layer allocations.", font=FONT_SMALL, fg="#4b5563").pack(anchor=tk.W, pady=(2, 0))

        # Main scrollable canvas container
        canvas = tk.Canvas(dlg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dlg, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, padx=14, pady=6)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mode Selection
        mode_frame = tk.LabelFrame(scrollable_frame, text="Simulation Mode", font=FONT_SECTION, padx=10, pady=8)
        mode_frame.pack(fill=tk.X, pady=(0, 8))
        self._add_info_btn_to_labelframe(mode_frame, self._show_default_case_info)

        sim_mode_var = tk.IntVar(value=2)  # Default to Custom Multi-Zone Mode

        rb_custom = tk.Radiobutton(mode_frame, text="Custom Multi-Zone Material Architecture", variable=sim_mode_var, value=2, font=FONT_SECTION)
        rb_custom.pack(anchor=tk.W)
        rb_p1 = tk.Radiobutton(mode_frame, text="Preset 1: Default Benchmark (10 Layers: 5x SS308L + 5x SS316L, 20 mm Wall)", variable=sim_mode_var, value=1)
        rb_p1.pack(anchor=tk.W)
        rb_p3 = tk.Radiobutton(mode_frame, text="Preset 3: 100 mm Tall Wall Build (50 Layers: 25x SS308L + 25x SS316L, 100 mm Wall)", variable=sim_mode_var, value=3)
        rb_p3.pack(anchor=tk.W)

        # 1. Substrate (Base Plate) Setup
        sub_frame = tk.LabelFrame(scrollable_frame, text="Substrate (Base Plate) Setup", font=FONT_SECTION, padx=10, pady=8)
        sub_frame.pack(fill=tk.X, pady=(0, 8))
        self._add_info_btn_to_labelframe(sub_frame, self._show_default_case_info)

        sf_grid = tk.Frame(sub_frame)
        sf_grid.pack(fill=tk.X)

        tk.Label(sf_grid, text="Substrate Material:", font=FONT_SMALL_BOLD).grid(row=0, column=0, sticky=tk.W, pady=3)
        sub_mat_combo = ttk.Combobox(sf_grid, values=mat_names, state="readonly", width=14)
        sub_mat_combo.set("SS316L")
        sub_mat_combo.grid(row=0, column=1, sticky=tk.W, padx=6, pady=3)

        tk.Label(sf_grid, text="Length X (mm):", font=FONT_SMALL).grid(row=1, column=0, sticky=tk.W, pady=2)
        lx_entry = ttk.Entry(sf_grid, width=10)
        lx_entry.insert(0, "150.0")
        lx_entry.grid(row=1, column=1, sticky=tk.W, padx=6, pady=2)

        tk.Label(sf_grid, text="Width Y (mm):", font=FONT_SMALL).grid(row=1, column=2, sticky=tk.W, padx=(12, 0), pady=2)
        ly_entry = ttk.Entry(sf_grid, width=10)
        ly_entry.insert(0, "150.0")
        ly_entry.grid(row=1, column=3, sticky=tk.W, padx=6, pady=2)

        tk.Label(sf_grid, text="Thickness Z (mm):", font=FONT_SMALL).grid(row=2, column=0, sticky=tk.W, pady=2)
        hz_entry = ttk.Entry(sf_grid, width=10)
        hz_entry.insert(0, "10.0")
        hz_entry.grid(row=2, column=1, sticky=tk.W, padx=6, pady=2)

        # 2. Deposit Wall Dimensions
        geom_frame = tk.LabelFrame(scrollable_frame, text="Deposit Wall Dimensions", font=FONT_SECTION, padx=10, pady=8)
        geom_frame.pack(fill=tk.X, pady=(0, 8))
        self._add_info_btn_to_labelframe(geom_frame, self._show_default_case_info)

        gf_grid = tk.Frame(geom_frame)
        gf_grid.pack(fill=tk.X)

        tk.Label(gf_grid, text="Single Bead Width Y (mm):", font=FONT_SMALL).grid(row=0, column=0, sticky=tk.W, pady=2)
        ty_entry = ttk.Entry(gf_grid, width=10)
        ty_entry.insert(0, "8.0")
        ty_entry.grid(row=0, column=1, sticky=tk.W, padx=6, pady=2)

        tk.Label(gf_grid, text="Single Layer Height Z (mm):", font=FONT_SMALL).grid(row=0, column=2, sticky=tk.W, padx=(12, 0), pady=2)
        hbead_entry = ttk.Entry(gf_grid, width=10)
        hbead_entry.insert(0, "2.0")
        hbead_entry.grid(row=0, column=3, sticky=tk.W, padx=6, pady=2)

        # 3. Multi-Zone / Multi-Material Allocation
        zones_frame = tk.LabelFrame(scrollable_frame, text="Multi-Zone Material Allocation", font=FONT_SECTION, padx=10, pady=8)
        zones_frame.pack(fill=tk.X, pady=(0, 8))
        self._add_info_btn_to_labelframe(zones_frame, self._show_default_case_info)

        num_zones_row = tk.Frame(zones_frame)
        num_zones_row.pack(fill=tk.X, pady=(0, 6))

        tk.Label(num_zones_row, text="Number of Material Zones (N):", font=FONT_SMALL_BOLD).pack(side=tk.LEFT)
        num_zones_var = tk.IntVar(value=2)
        num_zones_spin = ttk.Spinbox(num_zones_row, from_=1, to=8, width=5, textvariable=num_zones_var)
        num_zones_spin.pack(side=tk.LEFT, padx=8)

        # Container for dynamic zone rows
        zone_rows_container = tk.Frame(zones_frame)
        zone_rows_container.pack(fill=tk.X, pady=4)

        zone_widgets = []

        summary_label = tk.Label(zones_frame, text="Total Deposited Layers: 50 layers  |  Total Build Height: 100.0 mm", font=FONT_SMALL_BOLD, fg="#1d4ed8")
        summary_label.pack(anchor=tk.W, pady=(6, 2))

        def update_summary(*args):
            tot_layers = 0
            for w in zone_widgets:
                try:
                    tot_layers += int(w["cnt_var"].get())
                except ValueError:
                    pass
            try:
                hb = float(hbead_entry.get())
            except ValueError:
                hb = 2.0
            tot_h = tot_layers * hb
            summary_label.config(text=f"Total Deposited Layers: {tot_layers} layers  |  Total Build Height: {tot_h:.1f} mm")

        def rebuild_zone_rows():
            for widget in zone_rows_container.winfo_children():
                widget.destroy()
            zone_widgets.clear()

            n = max(1, min(8, num_zones_var.get()))
            default_layers = [25, 25, 20, 20, 15, 15, 10, 10]
            default_mats = ["SS308L", "SS316L", "ER70S", "AL1XXX", "SS308L", "SS316L", "ER70S", "AL1XXX"]

            for i in range(n):
                zr = tk.Frame(zone_rows_container, pady=3)
                zr.pack(fill=tk.X)

                tk.Label(zr, text=f"Zone {i+1}:", font=FONT_SMALL_BOLD, width=8, anchor=tk.W).pack(side=tk.LEFT)

                tk.Label(zr, text="Layers:").pack(side=tk.LEFT, padx=(4, 2))
                cnt_var = tk.StringVar(value=str(default_layers[i] if i < len(default_layers) else 10))
                cnt_var.trace_add("write", update_summary)
                cnt_entry = ttk.Entry(zr, textvariable=cnt_var, width=8)
                cnt_entry.pack(side=tk.LEFT, padx=4)

                tk.Label(zr, text="Material:").pack(side=tk.LEFT, padx=(8, 2))
                mat_var = tk.StringVar(value=default_mats[i] if i < len(default_mats) else "SS308L")
                mat_combo = ttk.Combobox(zr, textvariable=mat_var, values=mat_names, state="readonly", width=12)
                mat_combo.pack(side=tk.LEFT, padx=4)

                zone_widgets.append({
                    "cnt_var": cnt_var,
                    "cnt_entry": cnt_entry,
                    "mat_var": mat_var,
                    "mat_combo": mat_combo
                })

            update_summary()

        num_zones_var.trace_add("write", lambda *a: rebuild_zone_rows())
        hbead_entry.bind("<KeyRelease>", update_summary)
        rebuild_zone_rows()

        # 4. Thermal & Process Parameters
        proc_frame = tk.LabelFrame(scrollable_frame, text="Welding & Process Parameters", font=FONT_SECTION, padx=10, pady=8)
        proc_frame.pack(fill=tk.X, pady=(0, 10))
        self._add_info_btn_to_labelframe(proc_frame, self._show_default_case_info)

        pf_grid = tk.Frame(proc_frame)
        pf_grid.pack(fill=tk.X)

        tk.Label(pf_grid, text="Arc Power P (W):", font=FONT_SMALL).grid(row=0, column=0, sticky=tk.W, pady=2)
        power_entry = ttk.Entry(pf_grid, width=10)
        power_entry.insert(0, "2000.0")
        power_entry.grid(row=0, column=1, sticky=tk.W, padx=6, pady=2)

        tk.Label(pf_grid, text="Travel Speed v (mm/s):", font=FONT_SMALL).grid(row=0, column=2, sticky=tk.W, padx=(12, 0), pady=2)
        speed_entry = ttk.Entry(pf_grid, width=10)
        speed_entry.insert(0, "8.0")
        speed_entry.grid(row=0, column=3, sticky=tk.W, padx=6, pady=2)

        tk.Label(pf_grid, text="Inter-Pass Wait (s):", font=FONT_SMALL).grid(row=1, column=0, sticky=tk.W, pady=2)
        wait_entry = ttk.Entry(pf_grid, width=10)
        wait_entry.insert(0, "15.0")
        wait_entry.grid(row=1, column=1, sticky=tk.W, padx=6, pady=2)

        tk.Label(pf_grid, text="Ambient Temp (°C):", font=FONT_SMALL).grid(row=1, column=2, sticky=tk.W, padx=(12, 0), pady=2)
        tamb_entry = ttk.Entry(pf_grid, width=10)
        tamb_entry.insert(0, "25.0")
        tamb_entry.grid(row=1, column=3, sticky=tk.W, padx=6, pady=2)

        # Dialog Buttons
        btn_frame = tk.Frame(dlg, padx=14, pady=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        def on_run():
            chosen_mode = sim_mode_var.get()
            dlg.destroy()
            if chosen_mode == 1:
                self.material_base_var.set("SS316L")
                self.material_dep_var.set("SS308L")
                if hasattr(self, "us_layers_var"):
                    self.us_layers_var.set("10")
                self._run_simulation()
                return
            elif chosen_mode == 3:
                if hasattr(self, "us_layers_var"):
                    self.us_layers_var.set("50")
                if "lx" in self._param_vars:
                    self._param_vars["lx"][0].set("100.0")
                self._run_simulation()
                return

            # Extract custom parameters and apply dynamically to main application
            try:
                sub_mat = sub_mat_combo.get()
                self.material_base_var.set(sub_mat)

                if "lx" in self._param_vars:
                    self._param_vars["lx"][0].set(str(float(lx_entry.get())))
                if "ly_base" in self._param_vars:
                    self._param_vars["ly_base"][0].set(str(float(ly_entry.get())))
                if "hz_base" in self._param_vars:
                    self._param_vars["hz_base"][0].set(str(float(hz_entry.get())))
                if "ty_dep" in self._param_vars:
                    self._param_vars["ty_dep"][0].set(str(float(ty_entry.get())))
                if "h_bead" in self._param_vars:
                    self._param_vars["h_bead"][0].set(str(float(hbead_entry.get())))
                if "t_wait" in self._param_vars:
                    self._param_vars["t_wait"][0].set(str(float(wait_entry.get())))
                if "v_scan" in self._param_vars:
                    self._param_vars["v_scan"][0].set(str(float(speed_entry.get())))

                tot_layers = sum(int(zw["cnt_var"].get()) for zw in zone_widgets)
                if hasattr(self, "us_layers_var"):
                    self.us_layers_var.set(str(tot_layers))

                # Build and solve dynamically with EDPBuilder
                self._run_simulation()

            except Exception as ex:
                messagebox.showerror("Input Error", f"Please check numeric fields: {ex}")

        run_btn = tk.Button(
            btn_frame, text="Run Multi-Material Simulation",
            font=FONT_STATUS,
            bg=self.colors["button_primary"], fg=self.colors["button_primary_fg"],
            activebackground=self.colors["button_primary_active"],
            activeforeground=self.colors["button_primary_fg"],
            command=on_run, padx=12, pady=6
        )
        run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        cancel_btn = tk.Button(
            btn_frame, text="Cancel",
            font=FONT_BTN,
            bg=self.colors["button"], fg=self.colors["button_fg"],
            command=dlg.destroy, padx=12, pady=6
        )
        cancel_btn.pack(side=tk.LEFT)

    def _open_claude_paraview(self):
        """Open Multi-Material output waam.pvd in ParaView."""
        _here = Path(__file__).resolve().parent.parent
        pvd = self._claude_dir() / "waam_out" / "waam.pvd"
        if not pvd.exists():
            messagebox.showinfo("No Results", f"No waam.pvd found at:\n{pvd}\n\nPlease run the Multi-Material solver first.")
            return

        pv_path = shutil.which("paraview")
        if not pv_path:
            _app = "/Applications/ParaView.app/Contents/MacOS/paraview"
            if os.path.exists(_app):
                pv_path = _app
        if pv_path:
            subprocess.Popen([pv_path, str(pvd)])
            self._log(f"[PARAVIEW] Launched ParaView with: {pvd}")
        else:
            messagebox.showerror("ParaView Not Found", "ParaView binary could not be found.")

    def _open_claude_output(self):
        """Open the Multi-Material output folder."""
        _here = Path(__file__).resolve().parent.parent
        outdir = self._claude_dir() / "waam_out"
        outdir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(outdir)])
        else:
            subprocess.Popen(["xdg-open", str(outdir)])

    # Backward-compatibility aliases
    _run_multi_material_dialog = _run_claude_dialog
    _run_multi_mat_simulation = _run_claude_simulation
    _open_multi_mat_paraview = _open_claude_paraview
    _open_multi_mat_output = _open_claude_output

    def run(self):
        self.root.mainloop()


def main(preset_name=None):
    app = WAAMApp(initial_preset=preset_name)
    app.run()


if __name__ == "__main__":
    main()

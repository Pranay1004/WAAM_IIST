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
        "bg_input": "#edf2f7",
        "fg": "#1a202c",
        "fg_dim": "#718096",
        "accent": "#0d6efd",
        "accent2": "#198754",
        "warning": "#d97706",
        "error": "#dc3545",
        "border": "#cbd5e1",
        "button": "#e2e8f0",
        "button_fg": "#0f172a",
        "button_run": "#10b981",
        "button_run_fg": "#111827",
        "button_primary": "#2563eb",
        "button_primary_fg": "#111827",
        "button_primary_active": "#1d4ed8",
        "button_violet": "#7c3aed",
        "button_violet_fg": "#111827",
        "button_violet_active": "#6d28d9",
        "button_info": "#0284c7",
        "button_info_fg": "#111827",
        "tab_selected_bg": "#0d6efd",
        "tab_selected_fg": "#ffffff",
        "tab_unselected_bg": "#e2e8f0",
        "tab_unselected_fg": "#334155",
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
    "af", "ar", "bG", "cG",
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
        "k_spring_z": 1.0e4, "k_spring_xy": 1.0e3,
        "enable_radiation": True, "enable_mechanical": True,
        "enable_microstructure": True, "enable_boiling_cap": False,
        "enable_yield_truncation": False,
    },
    "SS308L on SS316L (UVA-WAAM Multi-Material)": {
        "material_base": "SS316L", "material_dep": "SS308L",
        "I_weld": 180.0, "U_weld": 22.0, "eta_arc": 0.82, "v_scan": 8.0,
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
        "lx": 150.0, "ly_base": 80.0, "hz_base": 12.0,
        "ty_dep": 7.0, "h_bead": 4.0, "remelt_ratio": 0.80, "n_layers": 20,
        "heat_source": "goldak",
    },
    "SS308L Stainless Steel (GMAW-WAAM)": {
        "material_base": "SS308L", "material_dep": "SS308L",
        "I_weld": 200.0, "U_weld": 24.0, "eta_arc": 0.80, "v_scan": 10.0,
        "lx": 200.0, "ly_base": 100.0, "hz_base": 10.0,
        "ty_dep": 8.0, "h_bead": 4.0, "remelt_ratio": 0.80, "n_layers": 26,
        "heat_source": "goldak",
    },
    "Ti-6Al-4V Titanium (Laser-DED / WAAM)": {
        "material_base": "Ti-6Al-4V", "material_dep": "Ti-6Al-4V",
        "I_weld": 1500.0, "U_weld": 1.0, "eta_arc": 0.40, "v_scan": 15.0,
        "lx": 100.0, "ly_base": 60.0, "hz_base": 8.0,
        "ty_dep": 5.0, "h_bead": 2.5, "remelt_ratio": 0.80, "n_layers": 30,
        "heat_source": "gaussian_vol",
    },
    "Pure Al (Al1xxx Wire-DED)": {
        "material_base": "Pure Al (Al1xxx)", "material_dep": "Pure Al (Al1xxx)",
        "I_weld": 140.0, "U_weld": 18.0, "eta_arc": 0.85, "v_scan": 12.0,
        "lx": 160.0, "ly_base": 90.0, "hz_base": 8.0,
        "ty_dep": 6.0, "h_bead": 3.0, "remelt_ratio": 0.80, "n_layers": 20,
        "heat_source": "goldak",
    },
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
        self.root.geometry("1520x880")
        self.root.minsize(1400, 780)

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
        self.registered_widgets = []

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
        self._refresh_embedded_viewport()
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._refresh_embedded_viewport())
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

        self.subtitle_label = tk.Label(
            self.title_frame,
            text="Coupled Thermal  |  Mechanics  |  Microstructure  |  OpenFOAM 2412 CFD",
            font=FONT_SUBTITLE
        )
        self.subtitle_label.pack(side=tk.TOP, anchor=tk.W)

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

        # Multi-Material menu
        claude_menu = tk.Menu(menubar, tearoff=0)
        claude_menu.add_command(label="Run Default Benchmark (10 Layers, 20 mm)", command=lambda: self._run_claude_simulation(preset=1))
        claude_menu.add_command(label="Run 100 mm Tall Wall Build (50 Layers, 100 mm)", command=lambda: self._run_claude_simulation(preset=3))
        claude_menu.add_command(label="Ultrasonic & Multi-Material Setup...", command=self._run_claude_dialog)
        claude_menu.add_separator()
        claude_menu.add_command(label="Open Multi-Material Results in ParaView", command=self._open_paraview)
        claude_menu.add_command(label="Open Multi-Material Output Directory", command=self._open_output)
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
            run_stop_frame, text="Run Simulation",
            font=FONT_BTN_RUN,
            command=self._run_simulation, padx=16, pady=6
        )
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.stop_btn = tk.Button(
            run_stop_frame, text="Stop Simulation",
            font=FONT_STATUS,
            command=self._stop_simulation, padx=12, pady=6,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.RIGHT, padx=(4, 0))

        # Primary Results & Post-Processing Bar (Dedicated row right below RUN SIMULATION)
        res_bar = tk.Frame(self.btn_bar)
        res_bar.pack(fill=tk.X, pady=(4, 2))
        self._reg(res_bar, "card_frame")

        self.view_results_btn = tk.Button(
            res_bar, text="View Results (Report)",
            font=FONT_STATUS,
            bg=self.colors["button_primary"], fg=self.colors["button_primary_fg"],
            activebackground=self.colors["button_primary_active"],
            activeforeground=self.colors["button_primary_fg"],
            command=self._open_report, padx=12, pady=5
        )
        self.view_results_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self.report_btn = self.view_results_btn

        self.pv_btn = tk.Button(
            res_bar, text="ParaView (Current Run)", font=FONT_BTN,
            command=self._open_paraview, padx=8, pady=5
        )
        self.pv_btn.pack(side=tk.LEFT, padx=2)

        self.out_btn = tk.Button(
            res_bar, text="Open Run Folder", font=FONT_BTN,
            command=self._open_output, padx=8, pady=5
        )
        self.out_btn.pack(side=tk.LEFT, padx=(2, 0))

        # Secondary Modeling & Viewport Bar
        sec_bar = tk.Frame(self.btn_bar)
        sec_bar.pack(fill=tk.X, pady=(2, 2))
        self._reg(sec_bar, "card_frame")

        self.gen_btn = tk.Button(
            sec_bar, text="Generate Simulation Script", font=FONT_BTN,
            command=self._generate_script, padx=6, pady=3
        )
        self.gen_btn.pack(side=tk.LEFT, padx=2)

        self.mesh3d_btn = tk.Button(
            sec_bar, text="Show 3D Tab", font=FONT_SECTION,
            bg=self.colors["button_info"], fg=self.colors["button_info_fg"],
            command=self._open_3d_mesh_viewer, padx=6, pady=3
        )
        self.mesh3d_btn.pack(side=tk.LEFT, padx=2)

        self.pv_gpu_btn = tk.Button(
            sec_bar, text="Open 3D Window", font=FONT_BTN,
            command=self._open_pyvista_window, padx=6, pady=3
        )
        self.pv_gpu_btn.pack(side=tk.LEFT, padx=2)

        self.foam_btn = tk.Button(
            sec_bar, text="Generate CFD Case", font=FONT_BTN,
            command=self._openfoam_dialog, padx=5, pady=3
        )
        self.foam_btn.pack(side=tk.LEFT, padx=2)

        # Multi-Material Solver Toolbar Row
        claude_bar = tk.Frame(self.btn_bar)
        claude_bar.pack(fill=tk.X, pady=(3, 2))
        self._reg(claude_bar, "card_frame")

        self.claude_btn = tk.Button(
            claude_bar, text="Multi-Material...", font=FONT_SECTION,
            bg=self.colors["button_violet"], fg=self.colors["button_violet_fg"],
            activebackground=self.colors["button_violet_active"],
            activeforeground=self.colors["button_violet_fg"],
            command=self._run_claude_dialog, padx=8, pady=3
        )
        self.claude_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.claude_100mm_btn = tk.Button(
            claude_bar, text="100 mm Build", font=FONT_BTN,
            command=lambda: self._run_claude_simulation(preset=3), padx=6, pady=3
        )
        self.claude_100mm_btn.pack(side=tk.LEFT, padx=2)

        self.claude_pv_btn = tk.Button(
            claude_bar, text="ParaView (Multi-Material)", font=FONT_BTN,
            command=self._open_paraview, padx=6, pady=3
        )
        self.claude_pv_btn.pack(side=tk.LEFT, padx=(2, 0))

        # Live Simulation Stage & Progress Tracker
        self.tracker_frame = tk.LabelFrame(
            self.right_panel, text="Simulation Progress",
            font=FONT_SECTION, padx=8, pady=4
        )
        self.tracker_frame.pack(fill=tk.X, padx=6, pady=(2, 4))
        self._reg(self.tracker_frame, "card_labelframe")

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

        self.console = scrolledtext.ScrolledText(
            self.console_frame,
            font=FONT_CONSOLE, wrap=tk.CHAR, state=tk.DISABLED
        )
        self.console.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
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

                with open(log_path, "w", encoding="utf-8") as log_f:
                    log_f.write(f"beamWeldFoam Solver Log\nCase: {foam_dir}\n\n")
                    for line in proc.stdout:
                        line = line.replace("\r", "")
                        while line and ord(line[0]) < 32:
                            line = line[1:]
                        log_f.write(line)
                        log_f.flush()
                        self.root.after(0, self._log, line)

                        # Track progress from solver output
                        if "Time =" in line:
                            self.root.after(0, lambda l=line.strip(): (
                                self.cfd_status_var.set(f"Solving: {l}"),
                            ))

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
        if not foam_dir or not os.path.exists(foam_dir):
            foam_dir = self._find_cfd_case_dir(self._get_active_or_latest_run_dir())

        if not os.path.exists(foam_dir):
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
        """Dedicated tab to view the waam_multilayer.edp default setup (Read-Only)."""
        self.default_case_tab = tk.Frame(self.notebook)
        self.notebook.add(self.default_case_tab, text="Default Case Setup")
        self._reg(self.default_case_tab, "card_frame")

        self.default_case_text = scrolledtext.ScrolledText(
            self.default_case_tab,
            font=FONT_MONO_LG, wrap=tk.NONE
        )
        self.default_case_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        
        # Load the default file contents (portable: framework DEFAULT.edp first)
        _here = Path(__file__).resolve().parent.parent
        default_file = _here / "DEFAULT.edp"
        if not default_file.exists():
            default_file = _here.parent / "DEFAULT.edp"
        try:
            with open(default_file, "r") as f:
                content = f.read()
            self.default_case_text.insert(tk.END, content)
        except Exception as e:
            self.default_case_text.insert(tk.END, f"Error loading {default_file}:\n{e}")
        
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
        desc.pack(fill=tk.X, pady=(0, 10))
        self._reg(desc, "label")

        # --- Ultrasonic & Geometry Parameters ---
        param_lf = tk.LabelFrame(content, text="Simulation Parameters", font=FONT_SECTION, padx=10, pady=8)
        param_lf.pack(fill=tk.X, pady=(0, 10))
        self._reg(param_lf, "card_labelframe")

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

        # --- Multi-Material Zones ---
        zone_lf = tk.LabelFrame(content, text="Multi-Material Zones (Layer-by-Layer Architecture)", font=FONT_SECTION, padx=10, pady=8)
        zone_lf.pack(fill=tk.X, pady=(0, 10))
        self._reg(zone_lf, "card_labelframe")

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

        # Hook trace on total layers, bead height, and remelt ratio to refresh zone ranges and summaries
        try:
            self.us_layers_var.trace_add("write", lambda *args: self._refresh_us_zones())
            self.us_height_var.trace_add("write", lambda *args: self._refresh_us_zones())
            self.us_remelt_var.trace_add("write", lambda *args: self._refresh_us_zones())
        except Exception:
            pass

        # Run Button
        action_box = tk.LabelFrame(content, text="Simulation Action", font=FONT_SECTION, padx=10, pady=8)
        action_box.pack(fill=tk.X, pady=(0, 10))
        self._reg(action_box, "card_labelframe")

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

    def _apply_preset(self, event=None):
        """Apply a named parameter preset to all relevant GUI fields."""
        name = self.preset_var.get()
        preset = PARAMETER_PRESETS.get(name)
        if not preset:
            return

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
                self._param_vars[key][0].set(str(val))
            self.params[key] = val

        self._update_mesh_calc()
        self._update_heat_calc()
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
        self.console.configure(
            bg=c["console_bg"], fg=c["console_fg"],
            insertbackground=c["console_fg"]
        )

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
                    highlightbackground=border,
                )
            except Exception:
                pass

        # Accent buttons keep their hue per theme (light saturated / dark pastel)
        # with dark (near-black) text as requested - never stuck white.
        try:
            self.view_results_btn.configure(
                bg=c["button_primary"], fg=c["button_primary_fg"],
                activebackground=c["button_primary_active"],
                activeforeground=c["button_primary_fg"],
            )
        except Exception:
            pass
        try:
            self.mesh3d_btn.configure(
                bg=c["button_info"], fg=c["button_info_fg"],
                activebackground=c["accent"],
                activeforeground=c["button_info_fg"],
            )
        except Exception:
            pass
        if hasattr(self, "claude_btn"):
            try:
                self.claude_btn.configure(
                    bg=c["button_violet"], fg=c["button_violet_fg"],
                    activebackground=c["button_violet_active"],
                    activeforeground=c["button_violet_fg"],
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
                    )
                except Exception:
                    pass

        self.run_btn.configure(
            bg=c["button_run"], fg=c["button_run_fg"],
            activebackground=c["accent2"], activeforeground=c["button_run_fg"],
        )
        self.stop_btn.configure(
            bg=c["error"], fg=c["button_primary_fg"] if self.current_theme == "dark" else "#ffffff",
            activebackground=border,
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
                    widget.configure(bg=c["bg_input"], fg=fg, insertbackground=fg, highlightbackground=border)
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

        if on_change:
            var.trace_add("write", lambda *args: on_change())
        else:
            if key in ("lx", "ly_base", "hz_base", "ty_dep", "h_bead", "remelt_ratio", "n_layers", "nx_base", "ny_base", "nz_base", "nx_dep", "ny_dep", "nz_dep"):
                var.trace_add("write", lambda *args: (self._update_mesh_calc(), self._update_geom_fusion_label()))
            if key in ("I_weld", "U_weld", "eta_arc", "v_scan", "ty_dep", "h_bead", "remelt_ratio"):
                var.trace_add("write", lambda *args: self._update_heat_calc())

        if unit:
            u_lbl = tk.Label(parent, text=unit, font=FONT_SMALL)
            u_lbl.grid(row=row, column=2, sticky=tk.W, padx=4)
            self._reg(u_lbl, "label_dim")

        self._param_vars[key] = (var, type_)
        return var

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
        self._make_param_row(lf1, "Substrate Length (Lₓ):", "lx", 0, "mm")
        self._make_param_row(lf1, "Substrate Width (W_sub):", "ly_base", 1, "mm")
        self._make_param_row(lf1, "Substrate Thickness (H_sub):", "hz_base", 2, "mm")

        lf2 = tk.LabelFrame(tab, text="WAAM Deposit Wall Geometry & Fusion Model", font=FONT_SECTION, padx=8, pady=4)
        lf2.pack(fill=tk.X, padx=8, pady=4)
        self._reg(lf2, "card_labelframe")
        self._make_param_row(lf2, "Single Bead Width (W_bead):", "ty_dep", 0, "mm")
        self._make_param_row(lf2, "First Layer Height (h₁):", "h_bead", 1, "mm")
        self._make_param_row(lf2, "Remelt / Fusion Ratio (η_remelt):", "remelt_ratio", 2, "[-]")
        self._make_param_row(lf2, "Deposited Layers Count (N_layer):", "n_layers", 3, "layers", type_="int")

        self.geom_fusion_info_lbl = tk.Label(
            lf2, text="—", font=FONT_SMALL_BOLD, fg="#2563eb", justify=tk.LEFT, anchor=tk.W
        )
        self.geom_fusion_info_lbl.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=8, pady=4)
        self._reg(self.geom_fusion_info_lbl, "label")
        self._update_geom_fusion_label()

    def _update_geom_fusion_label(self):
        """Update dynamic info label in Geometry tab for Layer 1, subsequent layers, and total build height."""
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
            nl = 10
            if "n_layers" in self._param_vars:
                raw_nl = self._param_vars["n_layers"][0].get().strip()
                if raw_nl:
                    nl = max(int(float(raw_nl)), 1)
            
            h_sub = h1 * remelt
            h_total = h1 if nl <= 1 else (h1 + (nl - 1) * h_sub)
            txt = (
                f"ℹ Fusion Profile: Layer 1 = {h1:.2f} mm | Subsequent = {h_sub:.2f} mm "
                f"({remelt*100.0:.0f}% height retention) | Total Deposit Height = {h_total:.2f} mm"
            )
            self.geom_fusion_info_lbl.config(text=txt)
        except Exception:
            pass

    def _build_process_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Process")
        self._reg(tab, "card_frame")

        lf1 = tk.LabelFrame(tab, text="Arc / Energy Process Parameters", font=FONT_SECTION, padx=8, pady=4)
        lf1.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf1, "card_labelframe")
        self._make_param_row(lf1, "Arc Current (I_arc):", "I_weld", 0, "A")
        self._make_param_row(lf1, "Arc Voltage (U_arc):", "U_weld", 1, "V")
        self._make_param_row(lf1, "Thermal Efficiency (η_arc):", "eta_arc", 2, "[-]")
        self._make_param_row(lf1, "Traverse Scan Speed (v_scan):", "v_scan", 3, "mm/s")
        self._make_param_row(lf1, "Inter-Pass Dwell (t_dwell):", "t_wait", 4, "s")

        # Calculated Heat Input & Energy Output Card
        self.heat_calc_frame = tk.LabelFrame(
            tab, text="Calculated Heat Input & Energy Output",
            font=FONT_SECTION, padx=8, pady=4
        )
        self.heat_calc_frame.pack(fill=tk.X, padx=8, pady=3)
        self._reg(self.heat_calc_frame, "card_labelframe")

        self.heat_calc_label = tk.Label(
            self.heat_calc_frame, text="—",
            font=FONT_MONO, justify=tk.LEFT, anchor=tk.W
        )
        self.heat_calc_label.pack(fill=tk.X, padx=4, pady=2)
        self._reg(self.heat_calc_label, "label")

        lf2 = tk.LabelFrame(tab, text="Heat Source Model", font=FONT_SECTION, padx=8, pady=4)
        lf2.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf2, "card_labelframe")

        sources = [
            ("goldak", "Goldak Double Ellipsoid (Arc / GMAW WAAM)"),
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

        # 2. Dissimilar Metal Interface Mismatch & Thermophysical Compatibility Card
        lf_mismatch = tk.LabelFrame(tab, text="Interfacial Thermophysical & CTE Compatibility", font=FONT_SECTION, padx=8, pady=4)
        lf_mismatch.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf_mismatch, "card_labelframe")

        self.dissimilar_info = tk.Text(lf_mismatch, height=5, width=60, font=FONT_MONO, state=tk.DISABLED, relief=tk.SOLID, bd=1)
        self.dissimilar_info.pack(fill=tk.X, padx=4, pady=4)
        self._reg(self.dissimilar_info, "entry")

        # 3. Physical Constants (Override)
        lf2 = tk.LabelFrame(tab, text="Physical Constants & Thermal Boundary Loss", font=FONT_SECTION, padx=8, pady=4)
        lf2.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf2, "card_labelframe")
        self._make_param_row(lf2, "Deposit Density (ρ):", "rho", 0, "kg/m³")
        self._make_param_row(lf2, "Solidus Temp (T_sol):", "T_sol", 1, "°C")
        self._make_param_row(lf2, "Liquidus Temp (T_liq):", "T_liq", 2, "°C")
        self._make_param_row(lf2, "Latent Heat of Fusion (L_f):", "latent_heat", 3, "J/kg")
        self._make_param_row(lf2, "Surface Emissivity (ε):", "emissivity", 4, "[-]")
        self._make_param_row(lf2, "Convection Coeff (h_conv):", "h_conv", 5, "W/(m²·K)")
        self._make_param_row(lf2, "Base Contact Loss (h_bot):", "h_bot", 6, "W/(m²·K)")
        self._make_param_row(lf2, "Mechanical Cutoff (T_mech):", "T_mech", 7, "°C")

        self._on_multi_material_change()

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
        cad_path = self.mesh_file_var.get().strip()
        if mesh_src == "cad_file" and cad_path and os.path.exists(cad_path):
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

    def _open_3d_webgl_studio(self):
        """Generate and launch the standalone Three.js GPU WebGL Studio inside the toolbox."""
        try:
            from gui.widgets.waam_3d_studio import launch_waam_3d_studio_window
            params = self._collect_params()
            if hasattr(self, "mesh_file_var") and self.mesh_file_var.get():
                mpath = self.mesh_file_var.get().strip()
                if os.path.exists(mpath):
                    params["gcode_file"] = mpath
            run_dir = self._get_active_or_latest_run_dir()
            vis_dir = self._fem_vis_dir(run_dir)
            os.makedirs(vis_dir, exist_ok=True)
            html_path = os.path.join(vis_dir, "waam_3d_studio.html")
            launch_waam_3d_studio_window(params, output_html_path=html_path)
            self._log(f"[3D STUDIO] [OK] Launched Hardware GPU WebGL 3D Studio: {html_path}\n")
        except Exception as e:
            self._log(f"[3D STUDIO] [ERROR] Could not launch 3D studio: {e}\n")
            messagebox.showerror("3D Studio Error", str(e))

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
                    f"Substrate Base Plate:    150.0 x 150.0 x 15.0 mm (dx = {dx_base:.2f} mm, dy = {dy_base:.2f} mm, dz = {dz_base:.2f} mm)\n"
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
        """Calculate and display real-time Heat Input and Power based on process inputs."""
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

            # 3. Volumetric Energy Density (J/mm^3)
            bead_area_mm2 = w_bead_mm * h_bead_mm
            e_vol = (h_net_j_mm / bead_area_mm2) if bead_area_mm2 > 0 else 0.0

            lines = [
                f"Net Linear Heat Input (H_net):     {h_net_j_mm:8.2f} J/mm   ({h_net_kj_mm:.3f} kJ/mm | {h_net_j_mm:.1f} kJ/m)",
                f"Gross Linear Heat Input (H_gross): {h_gross_j_mm:8.2f} J/mm   ({h_gross_kj_mm:.3f} kJ/mm)",
                f"Net Thermal Arc Power (P_net):     {p_net_kw:8.3f} kW     ({p_net_w:,.1f} W)",
                f"Gross Electrical Power (P_gross):  {p_gross_kw:8.3f} kW     ({p_gross_w:,.1f} W)",
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

                # Substrate plate is fixed to standard 150x150x15 mm
                if "lx" in self._param_vars:
                    self._param_vars["lx"][0].set("150.0")
                if "ly_base" in self._param_vars:
                    self._param_vars["ly_base"][0].set("150.0")
                if "hz_base" in self._param_vars:
                    self._param_vars["hz_base"][0].set("15.0")
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
                             f"Substrate 150x150x15 mm | {len(verts):,} Verts"
                    )

                self._update_mesh_calc()
                self._update_heat_calc()

                orig_str = f" | Original Size: {orig_bb.get('x', lx_mm):.1f} x {orig_bb.get('y', ly_mm):.1f} x {orig_bb.get('z', hz_mm):.1f} mm" if scale_applied < 0.999 else ""
                self._log(
                    f"\n[CAD-ENCLOSURE] Substrate Plate: 150.0 x 150.0 x 15.0 mm\n"
                    f"[CAD-ENCLOSURE] Build Enclosure Volume: 150.0 x 150.0 x 150.0 mm (3.375 L)\n"
                    f"[CAD-ENCLOSURE] Max Allowable Occupied Volume (75% Limit): 2.531 L (2,531,250 mm³)\n"
                    f"[CAD-IMPORT] Loaded CAD Model: {os.path.basename(path)} ({info['format']}){orig_str}\n"
                    f"[CAD-IMPORT] Full Path: {path}\n"
                    f"[CAD-IMPORT] Scale Applied: {scale_applied:.4f}x -> Occupied Volume: {v_occ:,.1f} mm³ ({vol_pct:.1f}% of enclosure)\n"
                    f"[CAD-IMPORT] Placed Bounding Box: Lx = {lx_mm:.2f} mm | Ly = {ly_mm:.2f} mm | Lz = {hz_mm:.2f} mm\n"
                    f"[CAD-IMPORT] Base Plate Placement: Centered at X=75.0, Y=75.0 mm | Resting on Substrate Top at Z=15.0 mm\n"
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

        self._log(f"[CAD-IMPORT] Target volume mesh size: ~{target_nodes} nodes (Gmsh 4.15 + OCCT volume pipeline)\n")
        self.status_var.set("Status: Meshing CAD...")

        def run_mesh():
            try:
                from mesh.intelligent_mesher import import_and_mesh_cad_intelligent
                outdir = "./waam_output"
                os.makedirs(outdir, exist_ok=True)
                out_mesh = os.path.join(outdir, f"imported_{Path(cad_path).stem}.mesh")
                result = import_and_mesh_cad_intelligent(cad_path, out_mesh, target_nodes=target_nodes, refine_thin_features=True)

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
        cad_path = self.mesh_file_var.get().strip()
        if not cad_path or not os.path.exists(cad_path):
            cad_path = getattr(self, "loaded_stl_path", "")
        if not cad_path or not os.path.exists(cad_path):
            cad_path = filedialog.askopenfilename(
                filetypes=[("3D CAD / STL", "*.stl *.step *.stp *.iges *.igs"), ("All", "*.*")],
                title="Select CAD Model to Slice",
                initialdir=str(Path.home() / "Desktop")
            )
            if not cad_path:
                return

        try:
            h_bead = float(self.slicer_h_var.get())
            w_bead = float(self.slicer_w_var.get())
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
            vis_dir = self._fem_vis_dir(run_dir)
            os.makedirs(vis_dir, exist_ok=True)

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

            # Auto-generate 3D WebGL Studio in visual_outputs for immediate visualization
            try:
                from gui.widgets.waam_3d_studio import generate_waam_3d_studio_html
                studio_html = os.path.join(vis_dir, "waam_3d_studio.html")
                params_tp = self._collect_params()
                params_tp["gcode_file"] = out_gcode
                params_tp["stl_file"] = cad_dest if os.path.exists(cad_dest) else cad_path
                generate_waam_3d_studio_html(params_tp, gcode_path=out_gcode, output_html_path=studio_html)
                self._log(f"[STUDIO] 3D WebGL Studio auto-generated at: {studio_html}\n")
            except Exception as se:
                self._log(f"[STUDIO] Notice: {se}\n")

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

        self.rad_var = tk.BooleanVar(value=True)
        cb1 = tk.Checkbutton(lf1, text="Surface Radiation (Stefan-Boltzmann ε·σ·(T⁴ - T_amb⁴))", variable=self.rad_var, font=FONT_SMALL)
        cb1.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb1, "check")

        self.latent_var = tk.BooleanVar(value=True)
        cb2 = tk.Checkbutton(lf1, text="Latent Heat of Fusion (Mushy zone apparent Cp)", variable=self.latent_var, font=FONT_SMALL)
        cb2.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb2, "check")

        self.mech_var = tk.BooleanVar(value=True)
        cb3 = tk.Checkbutton(lf1, text="Residual Stress & Substrate Distortion (Inherent Strain)", variable=self.mech_var, font=FONT_SMALL)
        cb3.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb3, "check")

        self.micro_var = tk.BooleanVar(value=True)
        cb4 = tk.Checkbutton(lf1, text="Microstructure & Vickers Hardness (G/R + JMAK kinetics)", variable=self.micro_var, font=FONT_SMALL)
        cb4.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb4, "check")

        self.vib_var = tk.BooleanVar(value=False)
        cb5 = tk.Checkbutton(lf1, text="Ultrasonic Vibration and Stress Relief (Modal + 20 kHz harmonic)", variable=self.vib_var, font=FONT_SMALL)
        cb5.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb5, "check")

        self.pool_var = tk.BooleanVar(value=False)
        cb6 = tk.Checkbutton(lf1, text="2D Marangoni Melt Pool Flow (Coupled Navier-Stokes)", variable=self.pool_var, font=FONT_SMALL)
        cb6.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb6, "check")

        self.foam_var = tk.BooleanVar(value=False)
        cb7 = tk.Checkbutton(lf1, text="Coupling to OpenFOAM 2412 (3D free-surface CFD)", variable=self.foam_var, font=FONT_SMALL)
        cb7.pack(anchor=tk.W, padx=4, pady=2)
        self._reg(cb7, "check")

        # Mechanical Residual Distortion & Inherent Strain Card
        lf_mech = tk.LabelFrame(tab, text="Mechanical Distortion & Inherent Strain Tensor Calibration", font=FONT_SECTION, padx=8, pady=4)
        lf_mech.pack(fill=tk.X, padx=8, pady=4)
        self._reg(lf_mech, "card_labelframe")

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

        # Anisotropic strain ratios and spring stiffnesses
        self._make_param_row(lf_mech, "Longitudinal Strain Ratio (χ_x):", "chi_x", 1, "[-]")
        self._make_param_row(lf_mech, "Transverse Strain Ratio (χ_y):", "chi_y", 2, "[-]")
        self._make_param_row(lf_mech, "Through-Thickness Ratio (χ_z):", "chi_z", 3, "[-]")
        self._make_param_row(lf_mech, "Isostatic Vertical Spring (k_z):", "k_spring_z", 4, "N/m³")
        self._make_param_row(lf_mech, "Isostatic Lateral Spring (k_xy):", "k_spring_xy", 5, "N/m³")

    def _build_solver_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Solver")
        self._reg(tab, "card_frame")

        lf1 = tk.LabelFrame(tab, text="Temporal Discretization", font=FONT_SECTION, padx=8, pady=4)
        lf1.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf1, "card_labelframe")
        self._make_param_row(lf1, "Time Scheme Weight (θ):", "theta", 0, "[-]")
        self._make_param_row(lf1, "Deposition Timestep (Δt_weld):", "dt_weld", 1, "s")
        self._make_param_row(lf1, "Inter-Pass Timestep (Δt_dwell):", "dt_wait", 2, "s")
        self._make_param_row(lf1, "Final Cooling Timestep (Δt_cool):", "dt_cool", 3, "s")
        self._make_param_row(lf1, "Cooling Steps Count (N_cool):", "n_cool_steps", 4, "steps", type_="int")

        lf2 = tk.LabelFrame(tab, text="Solver & Output Config", font=FONT_SECTION, padx=8, pady=4)
        lf2.pack(fill=tk.X, padx=8, pady=3)
        self._reg(lf2, "card_labelframe")
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
                if not val_str:
                    continue
                if type_ == "str":
                    params[key] = val_str
                elif type_ == "int":
                    params[key] = int(float(val_str))
                else:
                    params[key] = float(val_str)
                # Entry parsed OK: allow future warnings if it breaks again.
                warned = getattr(self, "_warned_param_keys", None)
                if warned is not None:
                    warned.discard(key)
            except Exception as e:
                # Warn once per key per session to avoid console spam
                # (_collect_params runs on every UI refresh / mesh calc).
                warned = getattr(self, "_warned_param_keys", None)
                if warned is None:
                    warned = set()
                    self._warned_param_keys = warned
                if key not in warned:
                    warned.add(key)
                    try:
                        self._log(f"[PARAM] [WARNING] Invalid entry for '{key}': {e}. Keeping {params.get(key)}.\n")
                    except Exception:
                        pass

        # Convert user-friendly millimeter entries into SI meters for FreeFEM solver
        for k in MM_DIMENSION_KEYS:
            if k in params:
                v = float(params[k])
                if v > 0.05:  # user entered mm (e.g. 200 mm -> 0.2 m, 8 mm -> 0.008 m)
                    params[k] = v / 1000.0

        for k in MM_SPEED_KEYS:
            if k in params:
                v = float(params[k])
                if v > 0.1:   # user entered mm/s (e.g. 10 mm/s -> 0.01 m/s)
                    params[k] = v / 1000.0

        params["heat_source"] = self.source_var.get()
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
        params["enable_vibration"] = self.vib_var.get()
        params["enable_meltpool"] = self.pool_var.get()
        params["enable_openfoam"] = self.foam_var.get()

        # --- Ultrasonic Tab Variables ---
        if hasattr(self, "us_freq_var"):
            try:
                params["ultrasonic_frequency"] = float(self.us_freq_var.get())
            except Exception:
                pass
        if hasattr(self, "us_layers_var"):
            try:
                nl = int(self.us_layers_var.get())
                params["num_layers"] = nl
                params["n_layers"] = nl
            except Exception:
                pass
        if hasattr(self, "us_width_var"):
            try:
                val = float(self.us_width_var.get())
                params["wBead"] = val / 1000.0 if val > 0.05 else val
                params["ty_dep"] = val / 1000.0 if val > 0.05 else val
            except Exception:
                pass
        if hasattr(self, "us_height_var"):
            try:
                val = float(self.us_height_var.get())
                params["hBead"] = val / 1000.0 if val > 0.05 else val
                params["h_bead"] = val / 1000.0 if val > 0.05 else val
                params["h_bead_first"] = val / 1000.0 if val > 0.05 else val
            except Exception:
                pass
        if hasattr(self, "us_remelt_var"):
            try:
                r_val = float(self.us_remelt_var.get())
                remelt = (r_val / 100.0) if r_val > 1.0 else r_val
                params["remelt_ratio"] = remelt
            except Exception:
                pass

        # Populate synchronized remelt and bead dimensions
        remelt = float(params.get("remelt_ratio", 0.80))
        h1 = float(params.get("h_bead_first", params.get("h_bead", 0.004)))
        params["h_bead_first"] = h1
        params["h_bead"] = h1
        params["remelt_ratio"] = remelt
        params["h_bead_subseq"] = h1 * remelt
        
        # Collect dynamic multi-material zones
        if hasattr(self, "us_zone_entries") and self.us_zone_entries:
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
                params["zones"] = zones_data
                params["zonea_layers"] = zones_data[0]["layers"]
                params["zonea_material"] = zones_data[0]["material"]
                if len(zones_data) > 1:
                    params["zoneb_layers"] = zones_data[1]["layers"]
                    params["zoneb_material"] = zones_data[1]["material"]
                params["material_dep"] = zones_data[0]["material"]
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
        self.console.config(state=tk.NORMAL)
        # Preserve single line breaks (previous version doubled every newline).
        text = msg if msg.endswith("\n") else msg + "\n"
        self.console.insert(tk.END, text)
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

            if preset_name == "DEFAULT" and not has_custom_cad:
                cand_paths = [
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DEFAULT.edp")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DEFAULT.edp")),
                ]
                _env_default = os.environ.get("WAAM_DEFAULT_EDP", "")
                if _env_default:
                    cand_paths.append(os.path.abspath(os.path.expanduser(_env_default)))
                default_edp = None
                for c in cand_paths:
                    if os.path.exists(c):
                        default_edp = c
                        break
                if default_edp and os.path.exists(default_edp):
                    import shutil
                    script_path = os.path.join(fem_dir, "waam_simulation.edp")
                    shutil.copy2(default_edp, script_path)
                    shutil.copy2(default_edp, os.path.join(fem_dir, "DEFAULT.edp"))

                    self._log(f"\n[PRESET] Applied preset: DEFAULT (Parametric Wall)\n")
                    self._log(f"[DEFAULT] Using authentic standalone FreeFEM++ script: {default_edp}\n")
                    self._log(f"[OUTPUT DIRECTORY] Directed to: {fem_dir}\n")
                    self._log(f"[VISUAL OUTPUTS]   Directed to: {vis_dir}\n")
                    self._log(f"[CODEGEN] [OK] FreeFEM++ script ready: {script_path}\n")

                    params_path = os.path.join(fem_dir, "parameters.json")
                    with open(params_path, 'w') as f:
                        json.dump(params, f, indent=2)
                    shutil.copy2(params_path, os.path.join(run_dir, "parameters.json"))
                    self._log(f"[CODEGEN] Parameters metadata saved: {params_path}\n")
                    self.status_var.set("● Status: Script Ready (DEFAULT.edp)")
                    return script_path

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

            # If UVA-WAAM SS308L preset is selected, directly load and run authentic uva_waam_ss308l_ss316.edp
            if preset_name in ["SS308L on SS316 (UVA-WAAM 20 kHz)", "SS308L on SS316 (UVA-WAAM 20kHz)", "SS308L_UVA_WAAM"] and not has_custom_cad:
                cand_paths = [
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uva_waam_ss308l_ss316.edp")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "uva_waam_ss308l_ss316.edp")),
                ]
                uva_edp = None
                for c in cand_paths:
                    if os.path.exists(c):
                        uva_edp = c
                        break
                if uva_edp and os.path.exists(uva_edp):
                    import shutil
                    script_path = os.path.join(fem_dir, "waam_simulation.edp")
                    shutil.copy2(uva_edp, script_path)
                    shutil.copy2(uva_edp, os.path.join(fem_dir, "uva_waam_ss308l_ss316.edp"))

                    self._log(f"\n[PRESET] Applied preset: SS308L on SS316 (UVA-WAAM 20 kHz)\n")
                    self._log(f"[UVA-WAAM] Using authentic FreeFEM++ script: {uva_edp}\n")
                    self._log(f"[OUTPUT DIRECTORY] Directed to: {fem_dir}\n")
                    self._log(f"[VISUAL OUTPUTS]   Directed to: {vis_dir}\n")
                    self._log(f"[CODEGEN] [OK] FreeFEM++ script ready: {script_path}\n")

                    params_path = os.path.join(fem_dir, "parameters.json")
                    with open(params_path, 'w') as f:
                        json.dump(params, f, indent=2)
                    shutil.copy2(params_path, os.path.join(run_dir, "parameters.json"))
                    self._log(f"[CODEGEN] Parameters metadata saved: {params_path}\n")
                    self.status_var.set("● Status: Script Ready (UVA)")
                    return script_path

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

            # Auto-generate 3D WebGL Studio into visual_outputs so it is always present in run folder
            try:
                from gui.widgets.waam_3d_studio import generate_waam_3d_studio_html
                studio_html = os.path.join(vis_dir, "waam_3d_studio.html")
                generate_waam_3d_studio_html(params, gcode_path=params.get("gcode_file"), output_html_path=studio_html)
            except Exception:
                pass

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

        def run_thread():
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
                    self.root.after(0, self._log, f"\n{'='*55}\n[OK] SIMULATION COMPLETED SUCCESSFULLY!\n{'='*55}\n")
                    self.root.after(0, self.status_var.set, "✓ Status: Completed")
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

                        self.root.after(0, self._log, f"[SUMMARY] Executive run summary saved: {os.path.join(target_run_dir, 'run_summary.json')}\n")
                        if plots:
                            self.root.after(0, self._log, f"[PLOTS] Generated {len(plots)} publication-quality 2D analysis plot(s) in: {os.path.join(target_run_dir, 'plots')}\n")
                        self.root.after(0, self._log, f"[PARAVIEW] Script created: {pv_scr}\n")

                        try:
                            from core.postprocess.output_manager import generate_enhanced_html_report
                            report_html = generate_enhanced_html_report(target_run_dir, p_final, summary, plots)
                            if report_html and os.path.exists(report_html):
                                self.root.after(0, self._log, f"[REPORT] Comprehensive engineering HTML report generated: {report_html}\n")
                                top_rep = os.path.join(case_dir, "report.html")
                                if top_rep != report_html:
                                    import shutil as _sh
                                    _sh.copy2(report_html, top_rep)
                            import shutil as _sh
                            top_log = os.path.join(case_dir, "sim.log")
                            if os.path.exists(log_file_path) and top_log != log_file_path:
                                _sh.copy2(log_file_path, top_log)
                                self.root.after(0, self._log, f"[LOG] Published full log to case root: {top_log}\n")
                        except Exception as rep_err:
                            self.root.after(0, self._log, f"[REPORT] Publishing note: {rep_err}\n")

                        self.root.after(500, self._log_reload)
                    except Exception as pe:
                        self.root.after(0, self._log, f"[POSTPROCESS] [WARNING] Post-processing note: {pe}\n")
                else:
                    self.root.after(0, self._log, f"\n[ERROR] FreeFEM exited with code {rc}\n")
                    self.root.after(0, self.status_var.set, f"✕ Status: Error (Exit {rc})")
            except Exception as e:
                self.root.after(0, self._log, f"\n[ERROR] Execution failure: {e}\n")
                self.root.after(0, self.status_var.set, "✕ Status: Execution Error")
            finally:
                self.running = False
                self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

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
        if self.process:
            try:
                self.process.terminate()
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

    def _run_claude_simulation(self, preset=1, custom_stdin=None):
        """Run the multi-material FreeFEM++ simulation with the specified preset or custom configuration."""
        if self.running:
            messagebox.showwarning("Simulation Running", "A simulation is already in progress. Please stop it first.")
            return

        _here = Path(__file__).resolve().parent.parent
        claude_dir = _here / "multi_materaial_claude"
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

                # Send preset choice or custom configuration sequence to stdin
                if preset == 2 and custom_stdin:
                    self.process.stdin.write(custom_stdin)
                else:
                    self.process.stdin.write(f"{preset}\n")
                self.process.stdin.flush()

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
            if chosen_mode in (1, 3):
                dlg.destroy()
                self._run_claude_simulation(preset=chosen_mode)
                return

            # Extract custom parameters
            try:
                sub_code = mat_map.get(sub_mat_combo.get(), 1)
                n_zones = len(zone_widgets)
                if n_zones < 1:
                    messagebox.showerror("Invalid Setup", "At least 1 material zone is required.")
                    return

                zone_specs = []
                for idx, zw in enumerate(zone_widgets):
                    cnt = int(zw["cnt_var"].get())
                    if cnt <= 0:
                        messagebox.showerror("Invalid Layer Count", f"Zone {idx+1} layer count must be > 0.")
                        return
                    mid = mat_map.get(zw["mat_var"].get(), 0)
                    zone_specs.append((cnt, mid))

                lx_val = float(lx_entry.get())
                ly_val = float(ly_entry.get())
                hz_val = float(hz_entry.get())
                ty_val = float(ty_entry.get())
                hb_val = float(hbead_entry.get())
                tw_val = float(wait_entry.get())
                ta_val = float(tamb_entry.get())
                p_val  = float(power_entry.get())
                v_val  = float(speed_entry.get())

                # Format exact stdin stream expected by cli.idp
                stdin_lines = [
                    "2",
                    str(sub_code),
                    str(n_zones),
                ]
                for cnt, mid in zone_specs:
                    stdin_lines.append(str(cnt))
                    stdin_lines.append(str(mid))

                stdin_lines.extend([
                    str(lx_val),
                    str(ly_val),
                    str(hz_val),
                    str(ty_val),
                    str(hb_val),
                    str(tw_val),
                    str(ta_val),
                    "1",  # direct power & speed mode
                    str(p_val),
                    str(v_val),
                ])
                custom_stdin_payload = "\n".join(stdin_lines) + "\n"

                dlg.destroy()
                self._run_claude_simulation(preset=2, custom_stdin=custom_stdin_payload)

            except ValueError as ex:
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
        pvd = _here / "multi_materaial_claude" / "waam_out" / "waam.pvd"
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
        outdir = _here / "multi_materaial_claude" / "waam_out"
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

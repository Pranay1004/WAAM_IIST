"""
gui.py - Tkinter desktop GUI for the WAAM slicer (no terminal needed).

Run:
    python waam_slicer/gui.py
or from the waam_lab folder:
    python -m waam_slicer.gui

Features:
- "Attach STL" file browser (fixes the `Main.stl` relative-path confusion:
  the dialog always returns an absolute path).
- All slicer params editable in a form (same defaults as main.py / machine
  nameplate: 800x800x400mm, 1000 mm/min max).
- One-click Slice + Generate, with live log, progress bar, mesh report,
  warnings, and stats.
- 2D layer preview (perimeter / infill / centerline in different colours).
- 3D "digital-twin-lite" replay: kinematic preview of the whole build,
  scrubbed with a slider (Phase-1 twin - see README / chat for full plan).
"""

import os
import sys
import time
import queue
import threading
import traceback

# Make imports work whether launched as `python waam_slicer/gui.py`
# (script dir on sys.path) or `python -m waam_slicer.gui` (package mode).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from waam_slicer.slicer_core import (
        load_mesh, slice_mesh, auto_place_mesh, apply_orientation,
        MeshValidationError, SlicingError,
    )
    from waam_slicer.toolpath import plan_all_layers, summarize_warnings, INFILL_STYLES
    from waam_slicer.gcode_gen import (
        WAAMParams, MachineLimits, generate_gcode, write_gcode_file,
        GCodeGenerationError, compute_job_stats, _format_time,
    )
except ImportError:  # fallback for plain `python waam_slicer/gui.py`
    from slicer_core import load_mesh, slice_mesh, auto_place_mesh, apply_orientation, MeshValidationError, SlicingError
    from toolpath import plan_all_layers, summarize_warnings, INFILL_STYLES
    from gcode_gen import (
        WAAMParams, MachineLimits, generate_gcode, write_gcode_file,
        GCodeGenerationError, compute_job_stats, _format_time,
    )

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


BEAD_COLORS = {"perimeter": "#1f77b4", "infill": "#ff7f0e", "centerline": "#d62728"}


class SlicerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WAAM Slicer - Vashishtha Research cell (DDCS V2.1 + Champ Pulse 500)")
        self.geometry("1180x760")
        self.minsize(1024, 680)

        self.log_q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.toolpaths = []
        self.layers = []
        self.mesh_report = None
        self.mesh = None
        self.last_gcode_path = ""
        self._preview_result = None   # background preview-load handoff (main thread polls it)
        self._preview_gen = 0         # invalidates stale preview loads

        self._build_widgets()
        self.after(120, self._poll_log_queue)

    # ---------------- layout ----------------
    def _build_widgets(self):
        # ---- top file bar ----
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="STL / mesh:").grid(row=0, column=0, sticky="w")
        self.stl_var = tk.StringVar()
        # pre-fill with repo's sample if present
        sample = os.path.join(_THIS_DIR, "Main.stl")
        if os.path.isfile(sample):
            self.stl_var.set(sample)
        ttk.Entry(top, textvariable=self.stl_var, width=62).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(top, text="Attach STL...", command=self.browse_stl).grid(row=0, column=2, padx=2)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Output G-code:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.out_var = tk.StringVar(value=os.path.join(os.getcwd(), "output.gcode"))
        ttk.Entry(top, textvariable=self.out_var, width=62).grid(row=1, column=1, padx=6, pady=(4, 0), sticky="ew")
        ttk.Button(top, text="Save as...", command=self.browse_out).grid(row=1, column=2, padx=2, pady=(4, 0))

        # ---- main split ----
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # left: params
        left = ttk.Frame(paned, padding=6)
        paned.add(left, weight=1)
        self._build_params(left)

        # right: notebook
        right = ttk.Frame(paned)
        paned.add(right, weight=3)
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_log = ttk.Frame(self.nb, padding=6)
        self.tab_orient = ttk.Frame(self.nb, padding=6)
        self.tab_2d = ttk.Frame(self.nb, padding=6)
        self.tab_3d = ttk.Frame(self.nb, padding=6)
        self.tab_empirical = ttk.Frame(self.nb, padding=6)
        self.nb.add(self.tab_log, text="Log / Report")
        self.nb.add(self.tab_orient, text="Orient & View")
        self.nb.add(self.tab_2d, text="2D Layer preview")
        self.nb.add(self.tab_3d, text="3D Twin replay (kinematic)")
        self.nb.add(self.tab_empirical, text="Empirical Values")

        logbar = ttk.Frame(self.tab_log)
        logbar.pack(fill=tk.X, side=tk.TOP)
        ttk.Label(logbar, text="Latest run is always at the bottom.").pack(side=tk.LEFT)
        ttk.Button(logbar, text="Clear log",
                   command=lambda: self.log_text.delete("1.0", tk.END)).pack(side=tk.RIGHT)
        self.log_text = tk.Text(self.tab_log, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb = ttk.Scrollbar(self.tab_log, command=self.log_text.yview)
        sb.pack(fill=tk.Y, side=tk.RIGHT)
        self.log_text.configure(yscrollcommand=sb.set)
        self.log("WAAM Slicer GUI ready. Attach an STL (or use the pre-filled Main.stl) and press Slice.")
        self.log("Tip: the earlier terminal error `string is not a file: Main.stl` just meant the")
        self.log("relative path was wrong - Main.stl lives in waam_slicer/. The GUI always uses absolute paths.")

        self._build_2d_tab()
        self._build_3d_tab()
        self._build_orient_tab()
        self._build_empirical_tab()

        # ---- bottom action bar ----
        bot = ttk.Frame(self, padding=8)
        bot.pack(side=tk.BOTTOM, fill=tk.X)
        self.run_btn = ttk.Button(bot, text="Slice + Generate G-code", command=self.start_job)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(bot, text="Open output folder", command=self.open_out_folder).pack(side=tk.LEFT, padx=8)
        ttk.Button(bot, text="Clear results", command=self._clear_results).pack(side=tk.LEFT)
        self.theme_var = tk.StringVar(value="light")
        self.theme_btn = ttk.Button(bot, text="Theme: light", command=self.toggle_theme)
        self.theme_btn.pack(side=tk.LEFT, padx=8)
        self.prog = ttk.Progressbar(bot, mode="indeterminate", length=180)
        self.prog.pack(side=tk.LEFT, padx=8)
        self.status_var = tk.StringVar(value="idle")
        ttk.Label(bot, textvariable=self.status_var).pack(side=tk.LEFT, padx=8)
        self.stats_var = tk.StringVar(value="")
        ttk.Label(bot, textvariable=self.stats_var, justify=tk.LEFT).pack(side=tk.RIGHT)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.apply_theme()

    def _on_close(self):
        """Tear down cleanly: a bare destroy()/Ctrl+C mid-draw can abort the
        interpreter inside matplotlib's C code (PyEval_RestoreThread fatal)."""
        try:
            if self._orient_after is not None:
                self.after_cancel(self._orient_after)
        except Exception:
            pass
        self._orient_after = None
        self._preview_gen += 1  # orphan any in-flight preview load
        for fig, _ax in self._mpl_axes():
            try:
                fig.clf()
            except Exception:
                pass
        try:
            self.quit()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def _param_row(self, parent, row, label, var, width=10, tip=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=row, column=1, sticky="ew", padx=(4, 2), pady=1)
        if tip:
            ttk.Label(parent, text=tip, foreground="gray", font=("TkDefaultFont", 8)).grid(
                row=row, column=2, sticky="w", padx=2)
        return e

    def _build_params(self, f):
        ttk.Label(f, text="Process parameters", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        grid = ttk.Frame(f)
        grid.pack(fill=tk.X, pady=4)
        grid.columnconfigure(1, weight=1)

        def dv(v):
            return tk.StringVar(value=str(v))

        self.p_layer_h = dv(2.0)
        self.p_bead_w = dv(5.0)
        self.p_perim = dv(2)
        self.p_travel = dv(400.0)
        self.p_weld = dv(150.0)
        self.p_wfs = tk.StringVar(value="")  # optional
        self.p_infill_step = dv(45.0)
        self.p_dwell = dv(0.0)
        self.p_lead = dv(0.0)
        self.p_arc_on = dv("M3")
        self.p_arc_off = dv("M5")
        self.p_wf_on = tk.StringVar(value="")
        self.p_wf_off = tk.StringVar(value="")
        self.p_xmin = dv(0.0); self.p_xmax = dv(800.0)
        self.p_ymin = dv(0.0); self.p_ymax = dv(800.0)
        self.p_zmin = dv(0.0); self.p_zmax = dv(400.0)
        self.p_maxfeed = dv(1000.0)
        self.p_stepmesh = dv(5.0)
        self.p_scale = dv(1.0)
        self.p_voltage = dv(28.0)
        self.p_current = dv(180.0)
        self.p_wire_eff = dv(0.8)
        self.p_wire_dia = dv(1.2)
        self.p_material = tk.StringVar(value="steel")

        rows = [
            ("Layer height (mm)", self.p_layer_h, "1.5-4 typical"),
            ("Bead width (mm)", self.p_bead_w, "4-8 typical"),
            ("Perimeters", self.p_perim, "auto-reduces if thin"),
            ("Travel speed (mm/min)", self.p_travel, ""),
            ("Weld speed (mm/min)", self.p_weld, ""),
            ("Wire feed (mm/min, opt)", self.p_wfs, "blank = off"),
            ("Infill angle step (deg)", self.p_infill_step, ""),
            ("Interlayer dwell (s)", self.p_dwell, "cooling pause"),
            ("Lead-in/out (mm)", self.p_lead, "0 = off"),
            ("Arc V (V)", self.p_voltage, "for heat input calc"),
            ("Arc I (A)", self.p_current, "for heat input calc"),
            ("Wire efficiency η", self.p_wire_eff, "0.7-0.9 GMAW"),
            ("Wire dia (mm)", self.p_wire_dia, "1.2 standard"),
            ("Arc ON M-code", self.p_arc_on, "confirm wiring!"),
            ("Arc OFF M-code", self.p_arc_off, ""),
            ("Wire ON M-code (opt)", self.p_wf_on, "blank = none"),
            ("Wire OFF M-code (opt)", self.p_wf_off, ""),
            ("X min", self.p_xmin, ""), ("X max", self.p_xmax, ""),
            ("Y min", self.p_ymin, ""), ("Y max", self.p_ymax, ""),
            ("Z min", self.p_zmin, ""), ("Z max", self.p_zmax, ""),
            ("Max feed (mm/min)", self.p_maxfeed, "nameplate 1000"),
            ("STEP mesh size (mm)", self.p_stepmesh, "STEP/IGES only"),
            ("Scale factor", self.p_scale, "1.0 = as-is"),
        ]
        for i, (lbl, var, tip) in enumerate(rows):
            self._param_row(grid, i, lbl, var, tip=tip)

        self.v_allow_damaged = tk.BooleanVar(value=False)
        self.v_skip_bounds = tk.BooleanVar(value=False)
        self.v_auto_place = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Auto-place part into envelope (center XY, Z=0)", variable=self.v_auto_place).pack(anchor="w")
        ttk.Checkbutton(f, text="Allow damaged mesh (not recommended)", variable=self.v_allow_damaged).pack(anchor="w")
        ttk.Checkbutton(f, text="Skip bounds check (simulation only!)", variable=self.v_skip_bounds).pack(anchor="w")
        irow = ttk.Frame(f)
        irow.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(irow, text="Infill style:").pack(side=tk.LEFT)
        self.p_infill_style = tk.StringVar(value="zigzag")
        ttk.Combobox(irow, textvariable=self.p_infill_style, values=list(INFILL_STYLES),
                     state="readonly", width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(irow, text="zigzag=connected  lines=separate strikes  contour=hollow",
                  foreground="gray", font=("TkDefaultFont", 8), wraplength=180).pack(side=tk.LEFT)
        mrow = ttk.Frame(f)
        mrow.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(mrow, text="Material:").pack(side=tk.LEFT)
        ttk.Combobox(mrow, textvariable=self.p_material,
                     values=["steel", "aluminium", "titanium"],
                     state="readonly", width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(mrow, text="steel  aluminium  titanium",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side=tk.LEFT)
        ttk.Label(f, text="Machine: DDCS V2.1, 800x800x400mm.\nArc defaults M3/M5 - confirm on a cold\ndry-run before any live weld (see README).",
                  foreground="#8a5a00", justify=tk.LEFT, wraplength=300).pack(anchor="w", pady=(8, 0))

        # orientation state (owned by the Orient & View tab, read by the slice job)
        self.o_rx = tk.DoubleVar(value=0.0)
        self.o_ry = tk.DoubleVar(value=0.0)
        self.o_rz = tk.DoubleVar(value=0.0)
        self.o_dx = tk.StringVar(value="0")
        self.o_dy = tk.StringVar(value="0")
        self.o_dz = tk.StringVar(value="0")
        self.preview_base = None       # raw mesh (scale=1) for the live view
        self.preview_path = ""         # path it was loaded from
        self._preview_face_idx = None  # decimated draw subset (None = draw all)
        self._preview_loading_for = ""  # path with a load in flight
        self._orient_after = None      # coalesced redraw handle

    def _build_2d_tab(self):
        ctl = ttk.Frame(self.tab_2d)
        ctl.pack(fill=tk.X)
        ttk.Label(ctl, text="Layer:").pack(side=tk.LEFT)
        self.layer_idx = tk.IntVar(value=0)
        self.layer_slider = ttk.Scale(ctl, from_=0, to=0, variable=self.layer_idx,
                                      command=lambda *_: self.draw_layer())
        self.layer_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.layer_lbl = ttk.Label(ctl, text="0 / 0")
        self.layer_lbl.pack(side=tk.LEFT)
        ttk.Button(ctl, text="Redraw", command=self.draw_layer).pack(side=tk.LEFT, padx=8)

        self.fig2d = Figure(figsize=(6, 4.6), dpi=100)
        self.ax2d = self.fig2d.add_subplot(111)
        self.ax2d.set_aspect("equal", adjustable="datalim")
        self.ax2d.set_xlabel("X (mm)"); self.ax2d.set_ylabel("Y (mm)")
        self.canvas2d = FigureCanvasTkAgg(self.fig2d, master=self.tab_2d)
        self.canvas2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_3d_tab(self):
        ctl = ttk.Frame(self.tab_3d)
        ctl.pack(fill=tk.X)
        ttk.Label(ctl, text="Show layers 0..").pack(side=tk.LEFT)
        self.twin_n = tk.IntVar(value=0)
        self.twin_slider = ttk.Scale(ctl, from_=0, to=0, variable=self.twin_n,
                                     command=lambda *_: self.draw_twin())
        self.twin_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.twin_lbl = ttk.Label(ctl, text="0 / 0")
        self.twin_lbl.pack(side=tk.LEFT)
        ttk.Button(ctl, text="Show all", command=self.show_all_twin).pack(side=tk.LEFT, padx=8)
        ttk.Label(ctl, text="Kinematic preview: ordered beads at true Z. NOT thermal/live.",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=8)

        self.fig3d = Figure(figsize=(6, 4.6), dpi=100)
        self.ax3d = self.fig3d.add_subplot(111, projection="3d")
        self.ax3d.set_xlabel("X (mm)"); self.ax3d.set_ylabel("Y (mm)"); self.ax3d.set_zlabel("Z (mm)")
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=self.tab_3d)
        self.canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------------- orient & live view ----------------
    def _build_orient_tab(self):
        ctrl = ttk.Frame(self.tab_orient, width=250)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        ctrl.pack_propagate(False)

        ttk.Label(ctrl, text="Rotate about centroid", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self._rot_row(ctrl, "RX", self.o_rx)
        self._rot_row(ctrl, "RY", self.o_ry)
        self._rot_row(ctrl, "RZ", self.o_rz)
        ttk.Button(ctrl, text="Reset orientation", command=self._reset_orientation).pack(anchor="w", pady=(2, 8))

        ttk.Label(ctrl, text="Position", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Checkbutton(ctrl, text="Auto-place in envelope", variable=self.v_auto_place,
                        command=self._schedule_orient_redraw).pack(anchor="w")
        shift = ttk.Frame(ctrl)
        shift.pack(fill=tk.X, pady=4)
        self._shift_row(shift, 0, "dX", self.o_dx)
        self._shift_row(shift, 1, "dY", self.o_dy)
        self._shift_row(shift, 2, "dZ", self.o_dz)
        ttk.Label(shift, text="Manual shift (mm, after auto-place)",
                  foreground="gray", font=("TkDefaultFont", 8)).grid(row=3, column=0, columnspan=3, sticky="w")

        ttk.Label(ctrl, text="Scale", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(6, 0))
        srow = ttk.Frame(ctrl)
        srow.pack(fill=tk.X)
        ttk.Entry(srow, textvariable=self.p_scale, width=8).pack(side=tk.LEFT)
        ttk.Label(srow, text="(same as params tab)", foreground="gray",
                  font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=4)

        self.fit_var = tk.StringVar(value="Attach a model to preview.")
        self.fit_lbl = ttk.Label(ctrl, textvariable=self.fit_var, justify=tk.LEFT, wraplength=240)
        self.fit_lbl.pack(anchor="w", pady=(8, 0))
        self.obounds_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self.obounds_var, justify=tk.LEFT, wraplength=240,
                  foreground="gray", font=("TkDefaultFont", 8)).pack(anchor="w")
        ttk.Button(ctrl, text="Refresh view", command=self._refresh_orient_view).pack(anchor="w", pady=(8, 0))
        ttk.Label(ctrl, text="Drag to orbit - scroll to zoom.\nWhat you see is what slices.",
                  foreground="gray", font=("TkDefaultFont", 8), justify=tk.LEFT).pack(anchor="w", pady=(6, 0))

        view = ttk.Frame(self.tab_orient)
        view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.figO = Figure(figsize=(6, 4.6), dpi=100)
        self.axO = self.figO.add_subplot(111, projection="3d")
        self.canvasO = FigureCanvasTkAgg(self.figO, master=view)
        self.canvasO.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        for v in (self.o_rx, self.o_ry, self.o_rz):
            v.trace_add("write", lambda *_: self._schedule_orient_redraw())
        for v in (self.o_dx, self.o_dy, self.o_dz, self.p_scale):
            v.trace_add("write", lambda *_: self._schedule_orient_redraw())
        self.v_auto_place.trace_add("write", lambda *_: self._schedule_orient_redraw())
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _rot_row(self, parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=label, width=3).pack(side=tk.LEFT)
        val = ttk.Label(row, text="0 deg", width=8)
        val.pack(side=tk.RIGHT)
        def on_move(*_):
            val.configure(text=f"{float(var.get()):.0f} deg")
        sc = ttk.Scale(row, from_=-180, to=180, variable=var, command=on_move)
        sc.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        var.trace_add("write", lambda *_: val.configure(text=f"{float(var.get()):.0f} deg"))

    def _shift_row(self, parent, r, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var, width=8).grid(row=r, column=1, padx=4, sticky="ew")
        ttk.Label(parent, text="mm", foreground="gray", font=("TkDefaultFont", 8)).grid(row=r, column=2, sticky="w")

    def _reset_orientation(self):
        self.o_rx.set(0.0); self.o_ry.set(0.0); self.o_rz.set(0.0)
        self.o_dx.set("0"); self.o_dy.set("0"); self.o_dz.set("0")
        self._schedule_orient_redraw()

    def _on_tab_changed(self, _event=None):
        try:
            if self.nb.select() == str(self.tab_orient):
                self._refresh_orient_view()
        except Exception:
            pass

    def _build_empirical_tab(self):
        """Build the Empirical Values tab with data from WAAM lecture slides."""
        import tkinter.font as tkfont
        nb = ttk.Notebook(self.tab_empirical)
        nb.pack(fill=tk.BOTH, expand=True)

        # --- Subtab 1: Metal Transfer Methods ---
        tab_transfer = ttk.Frame(nb, padding=8)
        nb.add(tab_transfer, text="Metal Transfer")
        ttk.Label(tab_transfer, text="MIG-WAAM Metal Transfer Methods & Deposit Rates",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 6))
        cols = ("Method", "Deposit Rate (kg/h)", "Current Range", "Notes")
        tree = ttk.Treeview(tab_transfer, columns=cols, show="headings", height=6)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=160 if c == "Method" else 130)
        data = [
            ("Short circuit / dip", "0.9 - 2.7", "Low (< 150A)", "Lowest heat input, good for thin walls"),
            ("Globular", "1.8 - 3.2", "Medium (150-200A)", "Gravity-dependent drop transfer"),
            ("Spray", "2.7 - 5.4", "High (> 200A)", "High deposition, directional"),
            ("Pulsed spray", "0.9 - 2.7", "Pulsed (varies)", "Controlled transfer, lower spatter"),
            ("Rotating spray", "6.3 - 13.5", "Very high (> 300A)", "Highest deposition rate"),
        ]
        for row in data:
            tree.insert("", tk.END, values=row)
        tree.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(tab_transfer, text="Source: IIT Goa WAAM Lecture slides (MIG-WAAM Deposition Process)",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(anchor="w")

        # --- Subtab 2: Geometric Formulas ---
        tab_geom = ttk.Frame(nb, padding=8)
        nb.add(tab_geom, text="Bead Geometry")
        ttk.Label(tab_geom, text="Single Bead Geometric Modeling (Parabolic Profile)",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 6))
        formulas = [
            "Bead profile:    y = h [1 - (2x/w)^2]",
            "Bead area:       A = (2/3) h w",
            "Volume flow:     Q = (pi d^2 / 4) x v_wire  (v_wire in m/min)",
            "Bead width:      w = 3 pi v_wire(m/min) d^2 / (8 h v_torch(mm/min))",
            "Deposition rate: D_r = A_wire x v_wire x rho  (gm/min)",
            "",
            "Multi-bead stepover (optimal):  p/2 = w/3",
            "Contact angle:   theta = arcsin(4 h w / (4 h^2 + w^2))",
            "Layer height:    h_n = 2 R cos(theta_n)",
        ]
        for f in formulas:
            ttk.Label(tab_geom, text=f, font=("TkDefaultFont", 10),
                      foreground="#222" if f else "white").pack(anchor="w", pady=1)
        ttk.Label(tab_geom, text="Source: IIT Goa WAAM Lecture slides (Geometric Modeling)",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(anchor="w", pady=(8, 0))

        # --- Subtab 3: Experimental Data ---
        tab_exp = ttk.Frame(nb, padding=8)
        nb.add(tab_exp, text="Experimental Data")
        ttk.Label(tab_exp, text="Experimental Parameter Sets (Champ Pulse 500)",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 6))
        cols2 = ("I (A)", "V (V)", "V_torch (mm/min)", "V_wire (m/min)", "Bead w (mm)", "Notes")
        tree2 = ttk.Treeview(tab_exp, columns=cols2, show="headings", height=6)
        for c in cols2:
            tree2.heading(c, text=c)
            tree2.column(c, width=90 if "mm" not in c else 110)
        exp_data = [
            ("70",  "25.9", "300", "0.0048",  "n/a", "Single bead, low current"),
            ("120", "19.0", "300", "0.0095",  "n/a", "Single bead"),
            ("140", "19.7", "400", "0.0119",  "n/a", "Higher speed"),
            ("200", "24.2", "300", "0.0199",  "n/a", "High current spray"),
        ]
        for row in exp_data:
            tree2.insert("", tk.END, values=row)
        tree2.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(tab_exp, text="Source: WhatsApp WAAM group (Pranay P, 14 Sep 2026)",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(anchor="w")

        # --- Subtab 4: Power & Heat Input ---
        tab_power = ttk.Frame(nb, padding=8)
        nb.add(tab_power, text="Power & Heat Input")
        ttk.Label(tab_power, text="Power Calculation (Rosenthal-based)",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 6))
        power_formulas = [
            "P = [rho A_w v_w (c_p (T_m - T_0) + L)",
            "     + rho A_p v_t (c_p (T_m - T_0) + L)] / eta",
            "",
            "Where:",
            "  rho   = wire density (kg/m^3)",
            "  A_w   = pi d_w^2 / 4  (wire cross-section, mm^2)",
            "  v_w   = wire feed speed (m/min)",
            "  c_p   = specific heat capacity (J/kg K)",
            "  T_m   = melting temperature (K)",
            "  T_0   = ambient temperature (K)",
            "  L     = latent heat of fusion (J/kg)",
            "  A_p   = penetration area = (2/3) p_n w  (mm^2)",
            "  v_t   = torch speed (mm/min)",
            "  eta   = process efficiency",
            "",
            "Linear heat input:  HI = eta V I / v_torch  (J/mm)",
            "Arc power:          P  = V x I  (W)",
            "Net heat power:     P_net = eta x V x I  (W)",
        ]
        for f in power_formulas:
            ttk.Label(tab_power, text=f, font=("TkDefaultFont", 10),
                      foreground="#222" if f else "white").pack(anchor="w", pady=1)
        ttk.Label(tab_power, text="Example: V_t=800, h=2, p_n=0.5, d_w=0.8, i=100A, rho=7800, eta=0.9 => P=731.5 W",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(anchor="w", pady=(8, 0))
        ttk.Label(tab_power, text="Source: IIT Goa WAAM Lecture slides (Slide 23)",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(anchor="w")

        # --- Subtab 5: Material Properties ---
        tab_mat = ttk.Frame(nb, padding=8)
        nb.add(tab_mat, text="Material Properties")
        ttk.Label(tab_mat, text="Wire/Thermal Properties by Material",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 6))
        cols3 = ("Material", "Density (kg/m^3)", "c_p (J/kg K)", "k (W/m K)", "T_m (K)", "L (J/kg)")
        tree3 = ttk.Treeview(tab_mat, columns=cols3, show="headings", height=4)
        for c in cols3:
            tree3.heading(c, text=c)
            tree3.column(c, width=100)
        mat_data = [
            ("Steel (ER70S-6)", "7850", "500", "50",  "1800", "2.7e5"),
            ("Aluminium (ER4043)", "2700", "900", "237", "933",  "3.9e5"),
            ("Titanium (Ti-6Al-4V)", "4510", "520", "22",  "1941", "3.6e5"),
        ]
        for row in mat_data:
            tree3.insert("", tk.END, values=row)
        tree3.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(tab_mat, text="Source: IIT Goa WAAM Lecture slides + standard references",
                  foreground="gray", font=("TkDefaultFont", 8)).pack(anchor="w")

    def _current_limits_or_default(self):
        try:
            return MachineLimits(
                x_min=float(self.p_xmin.get()), x_max=float(self.p_xmax.get()),
                y_min=float(self.p_ymin.get()), y_max=float(self.p_ymax.get()),
                z_min=float(self.p_zmin.get()), z_max=float(self.p_zmax.get()),
                max_feed_rate=float(self.p_maxfeed.get() or 1000.0),
            )
        except Exception:
            return MachineLimits()

    def _read_orient(self):
        """Rotation/shift/scale from the UI, validated. Raises GCodeGenerationError."""
        import math
        try:
            rx, ry, rz = float(self.o_rx.get()), float(self.o_ry.get()), float(self.o_rz.get())
            dx, dy, dz = float(self.o_dx.get() or 0), float(self.o_dy.get() or 0), float(self.o_dz.get() or 0)
            sc = float(self.p_scale.get() or 1.0)
        except ValueError:
            raise GCodeGenerationError("Orientation/shift/scale fields must be numbers.")
        for name, val in (("rotation", rx), ("rotation", ry), ("rotation", rz),
                          ("shift", dx), ("shift", dy), ("shift", dz), ("scale", sc)):
            if not math.isfinite(val):
                raise GCodeGenerationError(f"Orientation {name} must be finite, got {val}.")
        if sc <= 0:
            raise GCodeGenerationError(f"Scale must be positive, got {sc}.")
        return rx, ry, rz, dx, dy, dz, sc

    def _place_mesh(self, mesh, rx, ry, rz, dx, dy, dz, limits, auto_place, log):
        """Shared placement pipeline: rotate -> auto-place -> manual shift. Used by preview AND slice job."""
        if rx or ry or rz:
            apply_orientation(mesh, rx, ry, rz)
            log(f"  Rotated ({rx:g}, {ry:g}, {rz:g})deg about centroid; bounds={mesh.bounds.tolist()}")
        if auto_place:
            place_warns: list = []
            shift = auto_place_mesh(mesh, limits, _warn=place_warns)
            for w in place_warns:
                log(f"  [warn] {w}")
            if shift is not None:
                log(f"  Auto-placed by ({shift[0]:.1f}, {shift[1]:.1f}, {shift[2]:.1f})mm")
        if dx or dy or dz:
            mesh.apply_translation([dx, dy, dz])
            log(f"  Manual shift ({dx:g}, {dy:g}, {dz:g})mm; bounds={mesh.bounds.tolist()}")

    PREVIEW_MAX_FACES = 8000  # live view draws a subset; slicing always uses the full mesh

    def _refresh_orient_view(self):
        """Start a BACKGROUND load of the preview mesh (complex parts take
        seconds - never block the UI thread). Result is picked up by the
        polling loop via _preview_result."""
        path = self.stl_var.get().strip()
        if not path or not os.path.isfile(path):
            self.preview_base = None
            self.preview_path = ""
            self._preview_loading_for = ""
            self._draw_orient_empty("Attach a model file first.")
            return
        if path == self.preview_path and self.preview_base is not None:
            self._redraw_orient()
            return
        if path == getattr(self, "_preview_loading_for", ""):
            return  # load already in flight
        self._preview_loading_for = path
        self.preview_base = None  # mark stale so sliders can't draw the old part
        self._preview_gen += 1
        gen = self._preview_gen
        self._draw_orient_empty("Loading preview in background - watch here...")
        threading.Thread(target=self._preview_load_thread, args=(path, gen),
                         daemon=True).start()

    def _preview_load_thread(self, path, gen):
        try:
            base, _rep = load_mesh(path, strict=False, scale=1.0)
            n = len(base.faces)
            idx = None
            if n > self.PREVIEW_MAX_FACES:
                idx = list(np.random.default_rng(0).choice(
                    n, size=self.PREVIEW_MAX_FACES, replace=False))
            # hand off WITHOUT touching Tk (wrong thread) - the poll loop delivers it
            self._preview_result = (gen, path, base, idx, None)
        except Exception as e:  # noqa - surfaced in the fit label
            self._preview_result = (gen, path, None, None, str(e))

    def _preview_loaded(self, gen, path, base, idx, err):
        if gen != self._preview_gen:
            return  # superseded by a newer attach
        self._preview_loading_for = ""
        if err is not None or base is None:
            self.preview_base = None
            self.preview_path = ""
            self._preview_face_idx = None
            self._draw_orient_empty(f"Preview load failed: {err}")
            return
        self.preview_base = base
        self.preview_path = path
        self._preview_face_idx = idx
        self._redraw_orient()

    def _schedule_orient_redraw(self):
        if self._orient_after is not None:
            try:
                self.after_cancel(self._orient_after)
            except Exception:
                pass
        self._orient_after = self.after(120, self._redraw_orient)

    def _draw_orient_empty(self, msg):
        self.axO.clear()
        self.axO.set_title(msg)
        self.axO.set_xlabel("X (mm)"); self.axO.set_ylabel("Y (mm)"); self.axO.set_zlabel("Z (mm)")
        self._apply_mpl_theme(self.figO, self.axO)
        self.canvasO.draw()
        self.fit_var.set(msg)
        self.fit_lbl.configure(foreground="gray")
        self.obounds_var.set("")

    def _redraw_orient(self):
        self._orient_after = None
        if self.preview_base is None:
            if str(self.nb.select()) == str(self.tab_orient):
                self._refresh_orient_view()
            return
        try:
            self._redraw_orient_inner()
        except GCodeGenerationError as e:
            self.fit_var.set(str(e))
            self.fit_lbl.configure(foreground="red")
        except Exception as e:  # noqa - a preview must never take down the GUI
            self.fit_var.set(f"Preview error (slice job unaffected): {e}")
            self.fit_lbl.configure(foreground="red")

    def _redraw_orient_inner(self):
        try:
            rx, ry, rz, dx, dy, dz, sc = self._read_orient()
        except GCodeGenerationError as e:
            self.fit_var.set(str(e))
            self.fit_lbl.configure(foreground="red")
            return
        limits = self._current_limits_or_default()
        mesh = self.preview_base.copy()
        if abs(sc - 1.0) > 1e-12:
            mesh.apply_scale(sc)
        self._place_mesh(mesh, rx, ry, rz, dx, dy, dz, limits,
                         bool(self.v_auto_place.get()), log=lambda *_: None)

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self.axO.clear()
        idx = getattr(self, "_preview_face_idx", None)
        if idx is not None:
            try:
                sub = mesh.submesh([idx], append=True)
            except Exception:
                sub = mesh
        else:
            sub = mesh
        tris = sub.triangles
        if len(tris):
            self.axO.add_collection3d(Poly3DCollection(
                tris, facecolor="#4da3ff", edgecolor="none", alpha=0.8))
        self._draw_envelope(self.axO, limits)
        bmin, bmax = mesh.bounds
        self.axO.set_xlim(limits.x_min, limits.x_max)
        self.axO.set_ylim(limits.y_min, limits.y_max)
        self.axO.set_zlim(limits.z_min, limits.z_max)
        self.axO.set_xlabel("X (mm)"); self.axO.set_ylabel("Y (mm)"); self.axO.set_zlabel("Z (mm)")
        try:
            self.axO.set_box_aspect((
                limits.x_max - limits.x_min,
                limits.y_max - limits.y_min,
                limits.z_max - limits.z_min))
        except Exception:
            pass
        self.axO.set_title(f"rx={rx:g} ry={ry:g} rz={rz:g}  scale={sc:g}")
        self._apply_mpl_theme(self.figO, self.axO)
        self.canvasO.draw()

        import numpy as np
        size = bmax - bmin
        env = np.array([limits.x_max - limits.x_min, limits.y_max - limits.y_min,
                        limits.z_max - limits.z_min])
        self.obounds_var.set(
            f"Size {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm\n"
            f"Pos [{bmin[0]:.1f}..{bmax[0]:.1f}] [{bmin[1]:.1f}..{bmax[1]:.1f}] [{bmin[2]:.1f}..{bmax[2]:.1f}]"
            + (f"\nPreview decimated to {self.PREVIEW_MAX_FACES} faces (slice uses all {len(mesh.faces)})."
               if getattr(self, "_preview_face_idx", None) is not None else ""))
        if bool((size > env + 1e-6).any()):
            self.fit_var.set("TOO BIG for envelope - increase scale-down or split the part.")
            self.fit_lbl.configure(foreground="red")
        elif bool(((bmin < [limits.x_min, limits.y_min, limits.z_min]) |
                   (bmax > [limits.x_max, limits.y_max, limits.z_max])).any()):
            self.fit_var.set("Fits in size but sits OUTSIDE the envelope - enable auto-place.")
            self.fit_lbl.configure(foreground="orange")
        else:
            self.fit_var.set("FITS - ready to slice.")
            self.fit_lbl.configure(foreground="green")

    @staticmethod
    def _draw_envelope(ax, limits):
        x0, x1 = limits.x_min, limits.x_max
        y0, y1 = limits.y_min, limits.y_max
        z0, z1 = limits.z_min, limits.z_max
        corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                   (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            ax.plot([corners[a][0], corners[b][0]],
                    [corners[a][1], corners[b][1]],
                    [corners[a][2], corners[b][2]], color="gray", lw=0.8)
        # build plate
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z0] * 5, color="black", lw=1.2)

    # ---------------- file dialogs ----------------
    def browse_stl(self):
        p = filedialog.askopenfilename(title="Attach mesh file",
                                       filetypes=[("Mesh / CAD", "*.stl *.3mf *.obj *.step *.stp *.iges *.igs"),
                                                  ("All", "*.*")],
                                       initialdir=_THIS_DIR)
        if p:
            old = self.stl_var.get().strip()
            self.stl_var.set(p)
            self.log(f"Attached: {p}")
            self.preview_base = None  # force preview reload
            if p != old:
                # transforms belong to the previous file - a stale 0.1 scale on a
                # new part is exactly how silent mis-slices happen (cf. rocket_v2).
                self.p_scale.set("1.0")
                self._reset_orientation()
                self.log("Scale/orientation reset for the new file.")
                try:
                    self.nb.select(self.tab_orient)  # show the model immediately
                except Exception:
                    pass
            self._schedule_orient_redraw()

    def browse_out(self):
        p = filedialog.asksaveasfilename(title="Output G-code",
                                         defaultextension=".gcode",
                                         filetypes=[("G-code", "*.gcode *.tap *.nc"), ("All", "*.*")],
                                         initialfile="output.gcode")
        if p:
            self.out_var.set(p)

    def open_out_folder(self):
        import subprocess
        d = os.path.dirname(os.path.abspath(self.out_var.get())) or os.getcwd()
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", d])
            elif sys.platform.startswith("win"):
                os.startfile(d)  # noqa
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            messagebox.showinfo("Output folder", f"{d}\n({e})")

    # ---------------- theme ----------------
    THEMES = {
        "light": dict(bg="#f0f0f0", fg="#111111", field="#ffffff", text_bg="#ffffff",
                      text_fg="#111111", fig="#ffffff", ax="#ffffff", txt="#111111",
                      grid="#cccccc", accent="#0066cc"),
        "dark": dict(bg="#2b2b2b", fg="#e8e8e8", field="#3a3a3a", text_bg="#1e1e1e",
                     text_fg="#e8e8e8", fig="#1e1e1e", ax="#242424", txt="#e8e8e8",
                     grid="#555555", accent="#4da3ff"),
    }

    def toggle_theme(self):
        self.theme_var.set("dark" if self.theme_var.get() == "light" else "light")
        self.apply_theme()

    def apply_theme(self):
        name = self.theme_var.get()
        if hasattr(self, "theme_btn"):
            self.theme_btn.configure(text=f"Theme: {name}")
        t = self.THEMES.get(name, self.THEMES["light"])
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure(".", background=t["bg"], foreground=t["fg"])
        st.configure("TFrame", background=t["bg"])
        st.configure("TLabel", background=t["bg"], foreground=t["fg"])
        st.configure("TButton", background=t["field"], foreground=t["fg"])
        st.configure("TCheckbutton", background=t["bg"], foreground=t["fg"])
        st.configure("TEntry", fieldbackground=t["field"], foreground=t["fg"])
        st.configure("TCombobox", fieldbackground=t["field"], foreground=t["fg"],
                     background=t["field"])
        st.configure("TNotebook", background=t["bg"])
        st.configure("TNotebook.Tab", background=t["field"], foreground=t["fg"])
        st.configure("Horizontal.TProgressbar", background=t["accent"])
        if hasattr(self, "log_text"):
            self.log_text.configure(bg=t["text_bg"], fg=t["text_fg"],
                                    insertbackground=t["text_fg"])
        for fig, ax in self._mpl_axes():
            self._apply_mpl_theme(fig, ax)

    def _mpl_axes(self):
        out = []
        for fig_name, ax_name in (("fig2d", "ax2d"), ("fig3d", "ax3d"), ("figO", "axO")):
            fig = getattr(self, fig_name, None)
            ax = getattr(self, ax_name, None)
            if fig is not None and ax is not None:
                out.append((fig, ax))
        return out

    def _apply_mpl_theme(self, fig, ax):
        t = self.THEMES.get(self.theme_var.get(), self.THEMES["light"])
        try:
            fig.patch.set_facecolor(t["fig"])
            ax.set_facecolor(t["ax"])
            ax.title.set_color(t["txt"])
            ax.xaxis.label.set_color(t["txt"])
            ax.yaxis.label.set_color(t["txt"])
            try:
                ax.zaxis.label.set_color(t["txt"])
            except Exception:
                pass
            ax.tick_params(colors=t["txt"])
            for spine in ax.spines.values():
                spine.set_color(t["grid"])
            for pane in ("xaxis", "yaxis", "zaxis"):
                try:
                    getattr(ax, pane).pane.set_facecolor(t["ax"])
                    getattr(ax, pane).pane.set_edgecolor(t["grid"])
                except Exception:
                    pass
            fig.canvas.draw_idle()
        except Exception:
            pass

    # ---------------- results ----------------
    def _clear_results(self):
        """Wipe generated data (toolpaths, previews, stats) - also called on each new run."""
        self.toolpaths = []
        self.layers = []
        self.mesh = None
        self.mesh_report = None
        self.last_gcode_path = ""
        self.stats_var.set("")
        if hasattr(self, "status_var") and (self.worker is None or not self.worker.is_alive()):
            self.status_var.set("idle")
        for slider, var in ((getattr(self, "layer_slider", None), getattr(self, "layer_idx", None)),
                            (getattr(self, "twin_slider", None), getattr(self, "twin_n", None))):
            try:
                if slider is not None:
                    slider.configure(to=0)
                if var is not None:
                    var.set(0)
            except Exception:
                pass
        try:
            self.draw_layer()
        except Exception:
            pass
        try:
            self.draw_twin()
        except Exception:
            pass

    # ---------------- logging ----------------
    def log(self, msg):
        self.log_text.insert(tk.END, str(msg) + "\n")
        self.log_text.see(tk.END)

    def _poll_log_queue(self):
        try:
            while True:
                kind, payload = self.log_q.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "done":
                    self._on_job_done(payload)
                elif kind == "fail":
                    self._on_job_fail(payload)
        except queue.Empty:
            pass
        if self._preview_result is not None:  # background preview-load handoff (main thread)
            res, self._preview_result = self._preview_result, None
            try:
                self._preview_loaded(*res)
            except Exception:
                pass
        self.after(120, self._poll_log_queue)

    # ---------------- job ----------------
    def _read_params(self):
        def f(var, name, allow_blank_none=False, is_int=False):
            s = var.get().strip()
            if allow_blank_none and s == "":
                return None
            try:
                return int(s) if is_int else float(s)
            except ValueError:
                raise GCodeGenerationError(f"Parameter `{name}` is not a number: `{s}`.")

        bead_w = f(self.p_bead_w, "bead-width")
        return dict(
            layer_height=f(self.p_layer_h, "layer-height"),
            bead_width=bead_w,
            perimeters=f(self.p_perim, "perimeters", is_int=True),
            travel_speed=f(self.p_travel, "travel-speed"),
            weld_speed=f(self.p_weld, "weld-speed"),
            wire_feed_speed=f(self.p_wfs, "wire-feed-speed", allow_blank_none=True),
            infill_angle_step=f(self.p_infill_step, "infill-angle-step"),
            interlayer_dwell=f(self.p_dwell, "interlayer-dwell"),
            lead_in_out=f(self.p_lead, "lead-in-out"),
            arc_on=self.p_arc_on.get().strip() or "M3",
            arc_off=self.p_arc_off.get().strip() or "M5",
            wf_on=self.p_wf_on.get().strip() or None,
            wf_off=self.p_wf_off.get().strip() or None,
            x_min=f(self.p_xmin, "x-min"), x_max=f(self.p_xmax, "x-max"),
            y_min=f(self.p_ymin, "y-min"), y_max=f(self.p_ymax, "y-max"),
            z_min=f(self.p_zmin, "z-min"), z_max=f(self.p_zmax, "z-max"),
            max_feed=f(self.p_maxfeed, "max-feed-rate"),
            step_mesh=f(self.p_stepmesh, "step-mesh-size"),
            scale=f(self.p_scale, "scale"),
            infill_style=self._read_infill_style(),
            voltage=f(self.p_voltage, "voltage"),
            current=f(self.p_current, "current"),
            wire_efficiency=f(self.p_wire_eff, "wire-efficiency"),
            wire_diameter=f(self.p_wire_dia, "wire-diameter"),
            material=self.p_material.get().strip() or "steel",
            rot=(self._finite(float(self.o_rx.get()), "rotate-x"),
                 self._finite(float(self.o_ry.get()), "rotate-y"),
                 self._finite(float(self.o_rz.get()), "rotate-z")),
            shift=(self._finite(float(self.o_dx.get() or 0), "shift-x"),
                   self._finite(float(self.o_dy.get() or 0), "shift-y"),
                   self._finite(float(self.o_dz.get() or 0), "shift-z")),
        )

    def _read_infill_style(self):
        style = self.p_infill_style.get().strip()
        if style not in INFILL_STYLES:
            raise GCodeGenerationError(f"Infill style must be one of {list(INFILL_STYLES)}, got `{style}`.")
        return style

    @staticmethod
    def _finite(val, name):
        import math
        try:
            v = float(val)
        except (TypeError, ValueError):
            raise GCodeGenerationError(f"Parameter `{name}` is not a number: `{val}`.")
        if not math.isfinite(v):
            raise GCodeGenerationError(f"Parameter `{name}` must be finite, got `{val}`.")
        return v

    def start_job(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A slicing job is already running.")
            return
        stl = self.stl_var.get().strip()
        out = self.out_var.get().strip()
        if not stl:
            messagebox.showerror("No input", "Attach an STL file first (top-left button).")
            return
        if not os.path.isfile(stl):
            messagebox.showerror("File not found",
                                 f"`{stl}` does not exist.\nPick it with Attach STL (absolute path).")
            return
        try:
            params = self._read_params()
        except GCodeGenerationError as e:
            messagebox.showerror("Bad parameter", str(e))
            return
        self._clear_results()  # wipe stale previews/stats so a failed run can't masquerade as old data
        self.run_btn.configure(state=tk.DISABLED)
        self.prog.start(12)
        self.status_var.set("slicing...")
        self.log(f"\n=== Job started {time.strftime('%H:%M:%S')} ===")
        self.log(f"Input:  {stl}")
        self.log(f"Output: {out}")
        args = (stl, out, params,
                bool(self.v_allow_damaged.get()), bool(self.v_skip_bounds.get()),
                bool(self.v_auto_place.get()))
        self.worker = threading.Thread(target=self._job_thread, args=args, daemon=True)
        self.worker.start()

    def _job_thread(self, stl, out, p, allow_damaged, skip_bounds, auto_place=True):
        Q = self.log_q
        try:
            Q.put(("log", "[1/4] Loading mesh..."))
            mesh, report = load_mesh(stl, strict=not allow_damaged,
                                     step_mesh_size=p.get("step_mesh", 5.0),
                                     scale=p.get("scale", 1.0))
            Q.put(("log", f"  Watertight={report.watertight} bodies={report.body_count} "
                          f"non-manifold={report.non_manifold_edges} bounds={report.bounds}"))
            for w in report.warnings:
                Q.put(("log", f"  [warn] {w}"))

            limits = MachineLimits(
                x_min=p["x_min"], x_max=p["x_max"], y_min=p["y_min"],
                y_max=p["y_max"], z_min=p["z_min"], z_max=p["z_max"],
                max_feed_rate=p["max_feed"],
            )
            rx, ry, rz = p.get("rot", (0.0, 0.0, 0.0))
            dx, dy, dz = p.get("shift", (0.0, 0.0, 0.0))
            _msgs: list = []
            self._place_mesh(mesh, rx, ry, rz, dx, dy, dz, limits, auto_place,
                             log=_msgs.append)
            for m in _msgs:
                Q.put(("log", m))

            Q.put(("log", f"[2/4] Slicing at {p['layer_height']}mm..."))
            layers = slice_mesh(mesh, layer_height=p["layer_height"])
            Q.put(("log", f"  -> {len(layers)} layers"))

            Q.put(("log", f"[3/4] Planning toolpaths (bead={p['bead_width']}mm, {p['perimeters']} perim, infill={p.get('infill_style', 'zigzag')})..."))
            toolpaths = plan_all_layers(layers, bead_width=p["bead_width"],
                                        num_perimeters=p["perimeters"],
                                        infill_angle_step=p["infill_angle_step"],
                                        infill_style=p.get("infill_style", "zigzag"))
            n_beads = sum(len(tp.beads) for tp in toolpaths)
            Q.put(("log", f"  -> {n_beads} beads across {len(toolpaths)} layers"))
            warns = summarize_warnings(toolpaths)
            for w in warns[:15]:
                Q.put(("log", f"  [warn] {w}"))
            if len(warns) > 15:
                Q.put(("log", f"  ... and {len(warns) - 15} more"))

            Q.put(("log", "[4/4] Generating G-code (bounds-checked)..."))
            params = WAAMParams(
                travel_speed=p["travel_speed"], weld_speed=p["weld_speed"],
                wire_feed_speed=p["wire_feed_speed"],
                interlayer_dwell=p["interlayer_dwell"], lead_in_out=p["lead_in_out"],
                arc_on_mcode=p["arc_on"], arc_off_mcode=p["arc_off"],
                wire_feed_on_mcode=p["wf_on"], wire_feed_off_mcode=p["wf_off"],
            )
            stats = compute_job_stats(toolpaths, params,
                                      bead_width=p["bead_width"],
                                      layer_height=p["layer_height"],
                                      voltage=p.get("voltage", 28.0),
                                      current=p.get("current", 180.0),
                                      wire_efficiency=p.get("wire_efficiency", 0.8),
                                      wire_diameter=p.get("wire_diameter", 1.2),
                                      wire_feed_speed=p.get("wire_feed_speed"),
                                      material=p.get("material", "steel"))
            gcode = generate_gcode(toolpaths, params, bead_width=p["bead_width"],
                                   layer_height=p["layer_height"],
                                   header_comment=f"WAAM slice of {os.path.basename(stl)}",
                                   machine_limits=limits, skip_bounds_check=skip_bounds,
                                   job_stats=stats,
                                   voltage=p.get("voltage", 28.0),
                                   current=p.get("current", 180.0))
            write_gcode_file(gcode, out)
            Q.put(("done", dict(out=out, n_layers=len(layers), n_beads=n_beads,
                                n_warn=len(warns), n_lines=gcode.count("\n"),
                                report=report, mesh=mesh, toolpaths=toolpaths, layers=layers,
                                stats=stats)))
        except Exception as e:  # noqa - surface everything to the GUI
            Q.put(("fail", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))

    def _on_job_done(self, d):
        self.prog.stop()
        self.run_btn.configure(state=tk.NORMAL)
        self.status_var.set("done")
        self.mesh, self.mesh_report = d["mesh"], d["report"]
        self.toolpaths, self.layers = d["toolpaths"], d["layers"]
        self.last_gcode_path = d["out"]
        self.log(f"Done -> {d['out']} ({d['n_lines']} lines, {d['n_layers']} layers, {d['n_beads']} beads).")
        if d["n_warn"]:
            self.log(f"Remember: {d['n_warn']} warning(s) - review before hardware. See README dry-run steps.")
        else:
            self.log("No toolpath warnings.")
        self.stats_var.set(f"{d['n_layers']} layers | {d['n_beads']} beads | {d['n_lines']} lines")
        # Show full job stats in log
        st = d.get("stats")
        if st:
            self.log("")
            self.log("=" * 50)
            self.log("  JOB STATISTICS")
            self.log("=" * 50)
            self.log(f"  Weld path:       {st.weld_distance_mm:.0f} mm ({st.weld_distance_mm/1000:.2f} m)")
            self.log(f"  Weld time:       {_fmt_t(st.weld_time_s)}")
            self.log(f"  Travel time:     {_fmt_t(st.travel_time_s)}")
            self.log(f"  Arc dwell:       {_fmt_t(st.arc_dwell_s)}")
            if st.interlayer_dwell_s > 0:
                self.log(f"  Interlayer:      {_fmt_t(st.interlayer_dwell_s)}")
            self.log(f"  TOTAL BUILD:     {_fmt_t(st.total_build_time_s)}")
            self.log(f"  Wire from spool: {st.wire_length_m:.1f} m ({st.wire_mass_g:.0f} g)")
            self.log(f"  Deposited vol:   {st.deposited_volume_mm3:.0f} mm³ ({st.deposited_volume_mm3/1000:.1f} cm³)")
            self.log(f"  Mass dep. rate:  {st.mass_deposition_rate_kg_hr:.3f} kg/hr")
            self.log(f"  Arc power:       {st.arc_power_w:.0f} W  (V·I)")
            self.log(f"  Net heat power:  {st.net_heat_power_w:.0f} W  (η·V·I)")
            self.log(f"  Linear HI:       {st.heat_input_j_per_mm:.1f} J/mm")
            self.log(f"  Bead width (calc): {st.bead_width_calc:.2f} mm")
            self.log("=" * 50)
        # set sliders
        n = max(len(self.toolpaths) - 1, 0)
        self.layer_slider.configure(to=n)
        self.twin_slider.configure(to=n)
        self.twin_n.set(n)
        self.layer_idx.set(min(self.layer_idx.get(), n))
        self.draw_layer()
        self.draw_twin()
        self.nb.select(self.tab_2d)
        messagebox.showinfo("Done", f"G-code written:\n{d['out']}\n{d['n_layers']} layers, {d['n_beads']} beads.")

    def _on_job_fail(self, msg):
        self.prog.stop()
        self.run_btn.configure(state=tk.NORMAL)
        self.status_var.set("FAILED - see log")
        self.log(f"[FAILED]\n{msg}")
        self.nb.select(self.tab_log)
        first = msg.splitlines()[0] if msg else "Unknown error"
        messagebox.showerror("Slicing failed", first)

    # ---------------- previews ----------------
    def draw_layer(self):
        self.ax2d.clear()
        self.ax2d.set_aspect("equal", adjustable="datalim")
        self.ax2d.set_xlabel("X (mm)"); self.ax2d.set_ylabel("Y (mm)")
        if not self.toolpaths:
            self.ax2d.set_title("Run a slice to preview layers")
            self._apply_mpl_theme(self.fig2d, self.ax2d)
            self.canvas2d.draw()
            return
        i = int(round(float(self.layer_idx.get())))
        i = max(0, min(i, len(self.toolpaths) - 1))
        tp = self.toolpaths[i]
        self.layer_lbl.configure(text=f"{i} / {len(self.toolpaths) - 1}")
        for b in tp.beads:
            pts = np.asarray(b.points)
            if len(pts) < 2:
                continue
            self.ax2d.plot(pts[:, 0], pts[:, 1], color=BEAD_COLORS.get(b.kind, "k"), lw=1.2)
        # faint layer outline from sliced polygons
        if i < len(self.layers):
            for poly in self.layers[i].polygons:
                try:
                    x, y = poly.exterior.xy
                    self.ax2d.plot(x, y, color="gray", lw=0.6, alpha=0.5)
                except Exception:
                    pass
        # legend (proxy)
        for kind, c in BEAD_COLORS.items():
            self.ax2d.plot([], [], color=c, label=kind)
        self.ax2d.legend(fontsize=8, loc="best")
        self.ax2d.set_title(f"Layer {tp.index}  Z={tp.z:.2f}mm  ({len(tp.beads)} beads)")
        self.fig2d.tight_layout()
        self._apply_mpl_theme(self.fig2d, self.ax2d)
        self.canvas2d.draw()

    def draw_twin(self):
        self.ax3d.clear()
        self.ax3d.set_xlabel("X (mm)"); self.ax3d.set_ylabel("Y (mm)"); self.ax3d.set_zlabel("Z (mm)")
        if not self.toolpaths:
            self.ax3d.set_title("Kinematic twin - run a slice first")
            self._apply_mpl_theme(self.fig3d, self.ax3d)
            self.canvas3d.draw()
            return
        limits = self._current_limits_or_default()
        n = int(round(float(self.twin_n.get())))
        n = max(0, min(n, len(self.toolpaths) - 1))
        self.twin_lbl.configure(text=f"{n} / {len(self.toolpaths) - 1}")
        segs, cols = [], []
        for tp in self.toolpaths[:n + 1]:
            c = plt_color_for_z(tp.index, len(self.toolpaths))
            for b in tp.beads:
                pts = np.asarray(b.points)
                if len(pts) < 2:
                    continue
                z = np.full(len(pts), tp.z)
                xyz = np.column_stack([pts, z])
                segs.append(xyz[:-1])
                # per-segment colours not needed; one colour per layer band
                cols.append(c)
                # draw as line (cheap): plot each bead directly
                self.ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=c, lw=0.9)
        self.ax3d.set_title(f"Build replay: layers 0..{n}  (kinematic, no thermal)")
        try:
            self.ax3d.set_box_aspect((
                limits.x_max - limits.x_min,
                limits.y_max - limits.y_min,
                limits.z_max - limits.z_min))
        except Exception:
            pass
        self._apply_mpl_theme(self.fig3d, self.ax3d)
        self.canvas3d.draw()

    def show_all_twin(self):
        if self.toolpaths:
            self.twin_n.set(len(self.toolpaths) - 1)
            self.draw_twin()


def _fmt_t(seconds):
    """Format seconds as human-readable string for the GUI log."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h = m // 60
    m = m % 60
    return f"{h}h {m:02d}m {s:02d}s"


def plt_color_for_z(i, total):
    # simple blue->orange ramp by layer index
    t = i / max(total - 1, 1)
    r = 0.12 + 0.88 * t
    b = 0.70 - 0.55 * t
    return (r, 0.45, max(b, 0.08))


def main():
    import signal
    app = SlicerGUI()
    try:
        # Route Ctrl+C through the Tk event loop instead of raising
        # KeyboardInterrupt wherever the interpreter happens to be (often
        # inside matplotlib's C draw code, where it aborts the process).
        signal.signal(signal.SIGINT, lambda *_: app.after(0, app._on_close))
    except Exception:
        pass
    app.mainloop()


if __name__ == "__main__":
    main()

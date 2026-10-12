"""
Comprehensive test suite verifying:
1. Wall deposition pattern (bidirectional vs unidirectional) & G-code auto-update.
2. Default 5 baseplate center & wall build elevation history probe stations.
3. Ultrasonic physics gating (only evaluated when physics is enabled).
4. Dynamic 1-on-1 probe plotting from CSVs.
"""

import os
import sys
import tempfile
import csv
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codegen.edp_builder import EDPBuilder
from postprocess.plot_generator import generate_all_plots


def test_edp_deposition_pattern():
    print("[TEST 1] Testing deposition pattern in FreeFEM++ codegen...")
    # Bidirectional
    p_bi = EDPBuilder.default_params()
    p_bi["deposition_pattern"] = "bidirectional"
    edp_bi = EDPBuilder(p_bi).build()
    assert "bool isBidi = true;" in edp_bi, "Expected isBidi = true in bidirectional mode"
    assert "xTorch = (isBidi && (iL % 2 == 1))" in edp_bi, "Expected alternating xTorch kinematics"

    # Unidirectional
    p_uni = EDPBuilder.default_params()
    p_uni["deposition_pattern"] = "unidirectional"
    edp_uni = EDPBuilder(p_uni).build()
    assert "bool isBidi = false;" in edp_uni, "Expected isBidi = false in unidirectional mode"
    print("  -> Passed deposition pattern codegen tests.")


def test_ultrasonic_physics_gating():
    print("[TEST 2] Testing ultrasonic physics gating...")
    # Ultrasound disabled
    p_off = EDPBuilder.default_params()
    p_off["enable_ultrasonic"] = False
    p_off["ultrasonic_frequency"] = 20000.0  # Even if frequency field has a number
    edp_off = EDPBuilder(p_off).build()
    assert "real softenF = 1.0;" in edp_off, "Expected softenF = 1.0 when disabled"

    # Ultrasound enabled
    p_on = EDPBuilder.default_params()
    p_on["enable_ultrasonic"] = True
    p_on["ultrasonic_frequency"] = 20000.0
    edp_on = EDPBuilder(p_on).build()
    assert "real softenF = 0.8;" in edp_on, "Expected softenF = 0.8 when enabled"
    print("  -> Passed ultrasonic physics gating tests.")


def test_probe_stations_calculation():
    print("[TEST 3] Testing 5 default elevation probe stations calculation...")
    from gui.app import WAAMApp
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()  # headless
    app = WAAMApp(root)

    # Check default probe points string
    probe_txt = app._get_default_probe_points_text()
    lines = [ln.strip() for ln in probe_txt.splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(lines) == 5, f"Expected 5 default probe stations, got {len(lines)}"

    # Parse via _parse_probe_block
    params = app._collect_params()
    probes = params.get("probe_points", [])
    assert len(probes) == 5, f"Expected exactly 5 probe points in collected params, got {len(probes)}"

    # Check Z heights:
    # Station 1: Substrate mid depth
    # Station 2: Substrate surface
    # Station 3: 20% wall build height
    # Station 4: 50% wall build height
    # Station 5: 100% wall build height
    z1, z2, z3, z4, z5 = [p[2] for p in probes]
    assert z1 < z2, f"Station 1 Z ({z1}) must be below Station 2 Z ({z2})"
    assert z2 < z3, f"Station 2 Z ({z2}) must be below Station 3 Z ({z3})"
    assert z3 < z4, f"Station 3 Z ({z3}) must be below Station 4 Z ({z4})"
    assert z4 < z5, f"Station 4 Z ({z4}) must be below Station 5 Z ({z5})"

    # Verify X and Y are at baseplate center
    lx = params.get("lx", 0.15)
    ly = params.get("ly_base", 0.15)
    for idx, p in enumerate(probes):
        assert abs(p[0] - lx/2.0) < 1e-4, f"Probe {idx} X ({p[0]}) should be lx/2 ({lx/2.0})"
        assert abs(p[1] - ly/2.0) < 1e-4, f"Probe {idx} Y ({p[1]}) should be ly/2 ({ly/2.0})"

    # Test auto-update G-code
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = app._auto_update_wall_gcode(run_dir=tmp_dir)
        assert res is True
        gc_path = os.path.join(tmp_dir, "toolpath.gcode")
        tp_path = os.path.join(tmp_dir, "toolpath.txt")
        lt_path = os.path.join(tmp_dir, "layer_pass_times.txt")
        assert os.path.exists(gc_path), "toolpath.gcode was not created"
        assert os.path.exists(tp_path), "toolpath.txt was not created"
        assert os.path.exists(lt_path), "layer_pass_times.txt was not created"

        # Check toolpath.txt content
        with open(tp_path) as f:
            n_pts = int(f.readline().strip())
            assert n_pts > 0, "Expected non-empty toolpath.txt points"

        # Check layer pass times
        with open(lt_path) as f:
            n_lt = int(f.readline().strip())
            assert n_lt == params["n_layers"], f"Expected {params['n_layers']} layer times, got {n_lt}"

    root.destroy()
    print("  -> Passed probe stations and auto-update G-code tests.")


def test_dynamic_plot_generation():
    print("[TEST 4] Testing dynamic 1-on-1 probe plot generation...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create minimal synthetic simulation results with probe_points.csv
        plots_dir = os.path.join(tmp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        probe_csv = os.path.join(tmp_dir, "probe_points.csv")
        with open(probe_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["point_id", "time_s", "Temp_C", "Tpeak_C", "x_m", "y_m", "z_m", "sigM_MPa", "dispM_mm"])
            # Generate 5 probe points over 10 time steps
            z_levels = [0.005, 0.010, 0.014, 0.020, 0.030]
            for pid in range(5):
                z_val = z_levels[pid]
                for step in range(10):
                    t = step * 5.0
                    temp = 25.0 + 800.0 / (pid + 1) * (1.0 if step == (pid + 2) else 0.2)
                    writer.writerow([pid, t, temp, temp, 0.075, 0.075, z_val, 150.0, 0.5])

        # Run generate_all_plots
        generated = generate_all_plots(tmp_dir)

        # Verify master plot
        master_plot = os.path.join(plots_dir, "06_baseplate_center_thermal_cycle.png")
        assert os.path.exists(master_plot), "Master plot 06_baseplate_center_thermal_cycle.png was not created"
        assert master_plot in generated

        # Verify companion plot
        companion_plot = os.path.join(plots_dir, "06_elevation_probe_thermal_cycles.png")
        assert os.path.exists(companion_plot), "Companion plot was not created"
        assert companion_plot in generated

        # Verify all 5 individual 1-on-1 plots
        for pid in range(5):
            ind_plot = os.path.join(plots_dir, f"06_probe_pt{pid}_thermal_cycle.png")
            assert os.path.exists(ind_plot), f"Individual probe plot {ind_plot} was not created"
            assert ind_plot in generated

    print("  -> Passed dynamic 1-on-1 probe plot generation tests.")


def test_dimensioning_driver_strict_decoupling():
    print("[TEST 5] Testing dimensioning driver strict decoupling and 0-hardcoding...")
    from gui.app import WAAMApp
    import tkinter as tk

    app = WAAMApp()
    app.root.withdraw()

    # Case A: User sets 8 layers with h1=2.0 mm, remelt=0.80
    app._param_vars["h_bead"][0].set("2.0")
    app._param_vars["remelt_ratio"][0].set("0.80")
    app._param_vars["n_layers"][0].set("8")
    app._on_n_layers_key()

    assert app._layer_mode.get() == "n_layers"
    # Target height should auto-compute: 2.0 + 7 * 1.6 = 13.20 mm
    h_str = app._build_height_var.get().strip()
    assert abs(float(h_str) - 13.20) < 1e-2, f"Expected H_build = 13.20, got {h_str}"

    p8 = app._collect_params()
    assert p8["n_layers"] == 8, f"Expected 8 layers, got {p8['n_layers']}"
    edp8 = EDPBuilder(p8).build()
    assert "int  Nlayers    = 8;" in edp8, "FreeFEM EDP script should have Nlayers = 8"

    # Case B: User sets target height 15.0 mm with h1=2.0 mm, remelt=0.80
    app._build_height_var.set("15.0")
    app._on_build_height_key()

    assert app._layer_mode.get() == "height"
    # Layers should ceil to 10 (2.0 + 9 * 1.6 = 16.40 mm)
    nl_str = app._param_vars["n_layers"][0].get().strip()
    assert nl_str == "10", f"Expected n_layers = 10, got {nl_str}"

    p10 = app._collect_params()
    assert p10["n_layers"] == 10, f"Expected 10 layers, got {p10['n_layers']}"
    edp10 = EDPBuilder(p10).build()
    assert "int  Nlayers    = 10;" in edp10, "FreeFEM EDP script should have Nlayers = 10"

    # Case C: User sets 6 layers in n_layers mode
    app._param_vars["n_layers"][0].set("6")
    app._on_n_layers_key()
    assert app._layer_mode.get() == "n_layers"
    p6 = app._collect_params()
    assert p6["n_layers"] == 6, f"Expected 6 layers, got {p6['n_layers']}"
    edp6 = EDPBuilder(p6).build()
    assert "int  Nlayers    = 6;" in edp6, "FreeFEM EDP script should have Nlayers = 6"

    print("  -> Passed dimensioning driver strict decoupling and zero-hardcoding tests.")


if __name__ == "__main__":
    test_edp_deposition_pattern()
    test_ultrasonic_physics_gating()
    test_probe_stations_calculation()
    test_dynamic_plot_generation()
    test_dimensioning_driver_strict_decoupling()
    print("\nALL 5 TARGETED TESTS PASSED!")

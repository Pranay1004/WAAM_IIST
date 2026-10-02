"""Physics regression suite for the WAAM framework.

Run:  python3 tests/physics_suite.py
Exits non-zero if any check fails.  Dependencies: numpy only.

Covers the audit-fixed physics:
  1. Heat-source power normalization (Goldak, gaussian_vol, gaussian_surf,
     conical, uniform): quadrature of emitted/templated formulas == P.
  2. Latent-heat Cp-band encoding in material_database.json.
  3. VOF property lookup never raises (_get_metal_props).
  4. WRC-1992 ferrite-number monotonicity (308L > 316L).
  5. Generated builder .edp markers: exact weld time, epsSplit, boiling-cap
     accounting, wait/cool dt ramps, quiet-node clamp, quiet Cp full.
  6. Static presets (DEFAULT.edp, uva_*.edp) carry the same fixes.
  7. Module .edp files: conical Q0, solidification Tsol gate, complex
     harmonic solve in vibration_acoustic.
"""

import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB_PATH = os.path.join(ROOT, "core", "multimaterial", "material_database.json")
DEFAULT_EDP = os.path.join(ROOT, "generated_edps", "DEFAULT.edp")
UVA_EDP = os.path.join(ROOT, "generated_edps", "uva_waam_ss308l_ss316.edp")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + str(detail)) if detail and not ok else ""))


def read(path):
    with open(path) as f:
        return f.read()


def trapz(y, x):
    return float(np.trapezoid(np.asarray(y, float), np.asarray(x, float)))


# ----------------------------------------------------------------- 1. sources
def test_heat_source_normalization():
    print("1. Heat-source power normalization (quadrature == P)")
    P = 3840.0
    pi = math.pi

    # Goldak exactly as emitted by edp_builder (3*sqrt(3)*f*P/(pi*sqrt(pi)*abc))
    af, ar, bG, cG, ff, fr = 6.3e-3, 15.8e-3, 5.5e-3, 3.2e-3, 0.6, 1.4
    Q0f = 3 * math.sqrt(3) * ff * P / (pi * math.sqrt(pi) * af * bG * cG)
    Q0r = 3 * math.sqrt(3) * fr * P / (pi * math.sqrt(pi) * ar * bG * cG)
    x = np.linspace(-0.05, 0.05, 501)
    y = np.linspace(-0.03, 0.03, 301)
    z = np.linspace(-0.03, 0.03, 301)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    F = 0.5 * (1.0 + np.tanh(X / (0.1 * min(af, ar))))
    Q = (Q0f * np.exp(-3 * X ** 2 / af ** 2) * F
         + Q0r * np.exp(-3 * X ** 2 / ar ** 2) * (1 - F)) \
        * np.exp(-3 * Y ** 2 / bG ** 2) * np.exp(-3 * Z ** 2 / cG ** 2)
    dV = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
    I = float(Q.sum() * dV)
    check("goldak: int Q dV = P", abs(I / P - 1) < 0.01, "%.4f P" % (I / P))

    # gaussian_vol: Q0 = 3*sqrt(3)*P/(pi*sqrt(pi)*R^3)
    R = 0.005
    Q0 = 3 * math.sqrt(3) * P / (pi * math.sqrt(pi) * R ** 3)
    g = np.linspace(-6 * R, 6 * R, 301)
    Xg, Yg, Zg = np.meshgrid(g, g, g, indexing="ij")
    Ig = float((Q0 * np.exp(-3 * (Xg ** 2 + Yg ** 2 + Zg ** 2) / R ** 2)).sum()
               * (g[1] - g[0]) ** 3)
    check("gaussian_vol: int Q dV = P", abs(Ig / P - 1) < 0.01,
          "%.4f P" % (Ig / P))

    # gaussian_surf: plane integral of 3P/(pi R^2)*exp(-3 r^2/R^2) = P
    Rs = 0.004
    Q0s = 3 * P / (pi * Rs * Rs)
    r = np.linspace(-0.03, 0.03, 601)
    Xr, Yr = np.meshgrid(r, r, indexing="ij")
    plane = float((Q0s * np.exp(-3 * (Xr ** 2 + Yr ** 2) / Rs ** 2)).sum()
                  * (r[1] - r[0]) ** 2)
    check("gaussian_surf: plane integral = P", abs(plane / P - 1) < 0.01,
          "%.4f P" % (plane / P))

    # conical frustum: Q0 = 9P/(pi d S), plane integral at depth zeta
    rt, rb, d = 0.003, 0.001, 0.005
    S = rt * rt + rt * rb + rb * rb
    Q0c = 9 * P / (pi * d * S)
    zeta = np.linspace(0, d, 4000)
    Rl = rt + (rb - rt) * zeta / d
    Ic = trapz(Q0c * pi * Rl ** 2 / 3.0, zeta)
    check("conical: int Q dV = P", abs(Ic / P - 1) < 1e-4, "%.6f P" % (Ic / P))

    # uniform band: Q0 = P/(8abc) over box [-a,a]x[-b,b]x[-c,c]
    a, b, c = 0.01, 0.008, 0.004
    box = (P / (8 * a * b * c)) * (2 * a) * (2 * b) * (2 * c)
    check("uniform: int Q dV = P (analytic)", abs(box / P - 1) < 1e-12)

    # emitted constants from the builder match the formulas above
    from codegen.edp_builder import EDPBuilder
    p = EDPBuilder.default_params()
    p.update(heat_source="goldak")
    blk = EDPBuilder(p)._heat_source_block()
    m = re.search(r"Tsource = \(([-\d.e+]+)\*exp", blk)
    Q0f_b = 3 * math.sqrt(3) * p["ff"] * P / (pi * math.sqrt(pi) * p["af"]
                                              * p["bG"] * p["cG"])
    check("builder emits goldak Q0f (3*sqrt3 form)",
          bool(m) and abs(float(m.group(1)) - Q0f_b) / Q0f_b < 1e-6,
          m.group(1) if m else "not found")
    p.update(heat_source="gaussian_vol")
    blk = EDPBuilder(p)._heat_source_block()
    m = re.search(r"Tsource = ([-\d.e+]+) \* exp", blk)
    check("builder emits gaussian_vol Q0 (3*sqrt3 form)",
          bool(m) and abs(float(m.group(1)) - Q0) / Q0 < 1e-6,
          m.group(1) if m else "not found")
    p.update(heat_source="conical")
    blk = EDPBuilder(p)._heat_source_block()
    m = re.search(r"Tsource = \(zeta <= [0-9.e+-]+\+1.e-6\) \?\s*([-\d.e+]+)", blk)
    Q0expect = 9 * P / (pi * p["d_pen"] * (p["rt"] ** 2 + p["rt"] * p["rb"]
                                            + p["rb"] ** 2))
    check("builder emits conical Q0 = 9P/(pi*d*S)",
          bool(m) and abs(float(m.group(1)) - Q0expect) / Q0expect < 1e-6,
          m.group(1) if m else "not found")


# ----------------------------------------------------------- 2. latent heat
def test_latent_enthalpy():
    print("2. Latent-heat Cp-band encoding (trapz == base*dT + L)")
    db = json.load(open(DB_PATH))["alloys"]
    for name, a in db.items():
        T = a["specific_heat"]["T"]
        Cp = a["specific_heat"]["Cp"]
        Tsol, Tliq = a["solidus"], a["liquidus"]
        L = a["latent_heat"]
        below = [i for i, t in enumerate(T) if t < Tsol]
        band = [(t, c) for t, c in zip(T, Cp)
                if Tsol - 1e-6 <= t <= Tliq + 1e-6]
        if not below or len(band) < 2:
            check("%s: Cp band present" % name, False,
                  "below=%d band=%d" % (len(below), len(band)))
            continue
        base = Cp[below[-1]]
        tb = [t for t, _ in band]
        cb = [c for _, c in band]
        H = trapz(cb, tb)
        expect = base * (tb[-1] - tb[0]) + L
        rel = abs(H - expect) / max(L, 1.0)
        check("%s: band enthalpy = base*dT + L" % name, rel < 0.02,
              "%.0f vs %.0f J/kg (%.2f%%)" % (H, expect, rel * 100))


# --------------------------------------------------------- 3. VOF robustness
def test_vof_props():
    print("3. _get_metal_props never raises, finite outputs")
    from core.fluid.openfoam_coupling import _get_metal_props
    db = json.load(open(DB_PATH))["alloys"]
    names = list(db) + ["SS316L", "ss308l", "garbage-alloy", "", None]
    ok, detail = True, ""
    for n in names:
        try:
            pr = _get_metal_props({} if n is None else {"material": n})
            vals = [float(v) for v in pr.values() if isinstance(v, (int, float))]
            if not vals or not all(math.isfinite(v) for v in vals):
                ok, detail = False, "%r -> %r" % (n, pr)
                break
        except Exception as e:  # noqa: BLE001
            ok, detail = False, "%r raised %r" % (n, e)
            break
    check("all material keys resolve to finite props", ok, detail)


# ------------------------------------------------------- 4. WRC FN monotonic
def test_ferrite_number():
    print("4. WRC-1992 ferrite number (308L > 316L)")
    db = json.load(open(DB_PATH))["alloys"]

    def fn(comp):
        cr_eq = comp.get("Cr", 0) + comp.get("Mo", 0) + 1.5 * comp.get("Si", 0)
        ni_eq = (comp.get("Ni", 0) + 30 * comp.get("C", 0)
                 + 0.5 * comp.get("Mn", 0))
        ratio = cr_eq / max(ni_eq, 1e-9)
        return min(max(4.0 + 10.0 * (ratio - 1.4), 3.0), 12.0)

    f308 = fn(db["SS308L"]["composition"])
    f316 = fn(db["SS316L"]["composition"])
    check("FN(308L) > FN(316L)", f308 > f316, "%.2f vs %.2f" % (f308, f316))
    check("FN(308L) in plausible band", 5.0 <= f308 <= 10.0, "%.2f" % f308)
    check("FN(316L) in plausible band", 4.0 <= f316 <= 8.0, "%.2f" % f316)


# ----------------------------------------------- 5. generated-builder markers
def test_builder_markers():
    print("5. Builder-generated .edp markers")
    from codegen.edp_builder import EDPBuilder
    p = EDPBuilder.default_params()
    p.update(n_layers=2, nx_dep=8, ny_base=6, nx_base=8, nz_base=3, nz_dep=4,
             ny_dep=4, t_wait=2, dt_weld=1.0, n_cool_steps=2, dt_cool=5.0,
             output_dir="/tmp/")
    script = EDPBuilder(p).build()

    must = {
        "exact weld steps": r"NweldSteps = int\(tPassLayer/dtWeld \+ 0\.5\)",
        "boiling-cap energy accounting": r"EcapJ \+= int3d\(Th\)",
        "wait dt ramp": r"dtRampW",
        "cool dt ramp": r"dtRampC",
        "quiet-node clamp": r"if\(psiActive\[\]\[iv\] < 0\.5\) Temp\[\]\[iv\] = Tamb",
        "quiet nodes keep full Cp": r"CpEff\[\]\[iv\] = CpMat\(Tv, zv\);",
        "liquid-k boost": r"kLiqF",
        "EcapJ declared": r"real EcapJ",
    }
    for label, pat in must.items():
        check("builder script: " + label, re.search(pat, script) is not None,
              "pattern not found: " + pat)

    # Alloy kinetics generator (microstructure stage is not yet spliced into
    # build(); unit-test the generator directly so FN logic stays covered).
    b = EDPBuilder(p)
    ss_kinetics = b._get_alloy_kinetics_freefem_code("SS308L")
    check("builder script: FN kinetics (SS308L)",
          re.search(r"WRC-1992|Cr_eq|FN", ss_kinetics) is not None,
          "WRC-1992 FN block missing for SS308L")
    cs_kinetics = b._get_alloy_kinetics_freefem_code("ER70S-6")
    check("builder script: kinetics block (ER70S-6)",
          "kinetics" in cs_kinetics and "HVmap" in cs_kinetics,
          "carbon-steel kinetics/HV block missing")

    src = read(os.path.join(ROOT, "codegen", "edp_builder.py"))
    check("builder source: epsSplit relative rule",
          re.search(r"eps_split = max\(0\.1 \* min\([^)]+\), 0\.25 \* hx\)",
                    src) is not None, "eps_split rule not found")

    forbidden = {
        "real[int].length (FreeFEM 4.15 error)": r"real\[int\]\.length",
        "Cp multiplied by psiEff": r"CpEff\([^)]*\)\s*\*\s*psiEff",
        "CpEff = .*psiEff": r"CpEff\[\]\[iv\] = [^;\n]*psiEff",
        "hardcoded SS308L fallback silence": r"material_db\.get\(\s*[\"']SS308L",
    }
    for label, pat in forbidden.items():
        check("builder script: no " + label, re.search(pat, script) is None,
              "forbidden pattern present: " + pat)

    # heat-source power: emitted goldak constants integrate to P (defaults)
    m = re.search(r"real epsSplit = ([0-9.e+-]+);", script)
    hx = p["lx"] / p["nx_dep"]
    expect_eps = max(0.1 * min(p["af"], p["ar"]), 0.25 * hx)
    check("builder epsSplit value", bool(m)
          and abs(float(m.group(1)) - expect_eps) / expect_eps < 1e-6,
          m.group(1) if m else "not found")


# ------------------------------------------------------- 6. static preset files
def test_static_presets():
    print("6. Static presets (DEFAULT.edp, uva_*.edp) carry the fixes")
    files = {"DEFAULT": read(DEFAULT_EDP), "UVA": read(UVA_EDP)}
    must = {
        "quiet-node clamp": r"psiActive\[\]\[iv\] < 0\.5\) Temp\[\]\[iv\] = Tamb",
        "exact weld time": r"NweldSteps = int\(tPass\s*/\s*dtWeld\s*\+\s*0\.5\)",
        "wait dt ramp": r"dtRampW",
        "cool dt ramp": r"dtRampC",
        "boiling-cap accounting": r"EcapJ \+= int3d\(Th\)",
        "Goldak 3*sqrt3 (exact energy)": r"real Q0[fr] = 3\.0?\s*\*",
        "epsSplit relative":
            r"epsSplit = max\(0\.1\s*\*\s*min\(af,\s*ar\), 0\.25",
        "no hBot+radiation on floor": r"\(hBot\+hrad\)",
    }
    for fname, txt in files.items():
        for label, pat in must.items():
            if label.startswith("no "):
                check("%s: %s" % (fname, label),
                      re.search(pat, txt) is None, "pattern present: " + pat)
            else:
                check("%s: %s" % (fname, label),
                      re.search(pat, txt) is not None,
                      "pattern missing: " + pat)
        check("%s: no 6*sqrt3 energy doubling" % fname,
              re.search(r"Q0[fr]\s*=\s*6\.?\s*\*?\s*sqrt\(3", txt) is None,
              "Q0 uses 6*sqrt3 (2x power)")

    # DEFAULT-specific process calibration (melts at exact P=3840 W)
    check("DEFAULT: v_scan = 5 mm/s (melts at exact power)",
          re.search(r"real vscan\s*=\s*5\.e-3", files["DEFAULT"]) is not None,
          "vscan not recalibrated")
    # UVA-specific: no Pa-scale acoustic power
    check("UVA: PacouNom without *1e-6 (85 MPa, not 85 Pa)",
          re.search(r"PacouNom\s*=\s*[^;]*\*1\.?0?e-6", files["UVA"]) is None,
          "PacouNom still scaled to Pa")


# ------------------------------------------------------------ 7. module .edp
def test_modules():
    print("7. Core .edp module fixes")
    conical = read(os.path.join(ROOT, "core", "heat_sources",
                                "conical_frustum.edp"))
    check("conical_frustum: Q0 = 9P/(pi*d*S)",
          "9.*{{ Pweld }}/( {{ pi }}*dpen*" in conical
          and "Pweld }}/Vfrust" not in conical, "old P/V form still present")

    goldak = read(os.path.join(ROOT, "core", "heat_sources",
                               "goldak_double_ellipsoid.edp"))
    check("goldak module: 3*sqrt3 form (exact energy)",
          re.search(r"real Q0f = 3\.\*sqrt\(3\.\)", goldak) is not None
          and re.search(r"real Q0f = 6\.", goldak) is None,
          "module still uses 6*sqrt3 (2x power)")
    gvol = read(os.path.join(ROOT, "core", "heat_sources",
                             "gaussian_volumetric.edp"))
    check("gaussian_volumetric module: 3*sqrt3 form",
          re.search(r"real Q0gauss = 3\.\*sqrt\(3\.\)", gvol) is not None
          and re.search(r"real Q0gauss = 6\.", gvol) is None,
          "module still uses 6*sqrt3 (2x power)")

    sol = read(os.path.join(ROOT, "core", "microstructure",
                            "solidification_phases.edp"))
    check("solidification: melt gate at Tsol (mushy zone included)",
          re.search(r"if\(Tp > Tsol &&", sol) is not None
          and re.search(r"if\(Tp > Tliq &&", sol) is None,
          "gate still at Tliq")

    vib = read(os.path.join(ROOT, "core", "acoustic",
                            "vibration_acoustic.edp"))
    check("vibration: complex harmonic solve (damping included)",
          "fespace Vhc" in vib and "ii*omegaE" in vib
          and "i*omega*alpha*M" in vib, "no complex solve found")
    check("vibration: force normalised to ampForce",
          "ampForce/zPatch" in vib, "patch load not normalised")
    check("vibration: VSR labelled estimate",
          "[ESTIMATE]" in vib, "VSR not labelled")


# ----------------------------------------------------------------------- main
def main():
    test_heat_source_normalization()
    test_latent_enthalpy()
    test_vof_props()
    test_ferrite_number()
    test_builder_markers()
    test_static_presets()
    test_modules()

    failed = [r for r in RESULTS if not r[1]]
    print("\n%d/%d checks passed" % (len(RESULTS) - len(failed), len(RESULTS)))
    if failed:
        print("FAILED:")
        for name, _, detail in failed:
            print("  - %s%s" % (name, ("  -- " + detail) if detail else ""))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

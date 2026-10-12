import os

# See scratch/create_0.py: research paths are environment-overridable.
WAAM_SCRATCH_ROOT = os.environ.get("WAAM_SCRATCH_ROOT", "/Users/pandeyji/Desktop/WAAM")
WAAM_SOLVER_SRC = os.path.join(WAAM_SCRATCH_ROOT, "waamFoam")
WAAM_CASE_DIR = os.path.join(WAAM_SCRATCH_ROOT, "waamCase")

readWAAMProperties_H = """//- Read WAAM Multiphysics Properties
Info<< "Reading waamProperties\\n" << endl;

IOdictionary waamProperties
(
    IOobject
    (
        "waamProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

// Thermal properties
dimensionedScalar T_sol("T_sol", dimTemperature, waamProperties.lookupOrDefault<scalar>("T_sol", 1698.0));
dimensionedScalar T_liq("T_liq", dimTemperature, waamProperties.lookupOrDefault<scalar>("T_liq", 1778.0));
dimensionedScalar L_fusion("L_fusion", dimEnergy/dimMass, waamProperties.lookupOrDefault<scalar>("L_fusion", 2.7e5));
dimensionedScalar Cp_metal("Cp_metal", dimEnergy/dimMass/dimTemperature, waamProperties.lookupOrDefault<scalar>("Cp_metal", 750.0));
dimensionedScalar Cp_gas("Cp_gas", dimEnergy/dimMass/dimTemperature, waamProperties.lookupOrDefault<scalar>("Cp_gas", 1000.0));
dimensionedScalar k_metal("k_metal", dimPower/dimLength/dimTemperature, waamProperties.lookupOrDefault<scalar>("k_metal", 25.0));
dimensionedScalar k_gas("k_gas", dimPower/dimLength/dimTemperature, waamProperties.lookupOrDefault<scalar>("k_gas", 0.05));
dimensionedScalar beta_T("beta_T", dimless/dimTemperature, waamProperties.lookupOrDefault<scalar>("beta_T", 1.6e-5));
dimensionedScalar T_ref("T_ref", dimTemperature, waamProperties.lookupOrDefault<scalar>("T_ref", 300.0));

// Hydrodynamic & Porosity properties
dimensionedScalar C_mush("C_mush", dimDensity/dimTime, waamProperties.lookupOrDefault<scalar>("C_mush", 1.0e7));

// Marangoni Surface Tension (Fe-S Pseudo-binary System)
dimensionedScalar gamma_m0("gamma_m0", dimForce/dimLength, waamProperties.lookupOrDefault<scalar>("gamma_m0", 1.943));
dimensionedScalar A_gamma("A_gamma", dimForce/dimLength/dimTemperature, waamProperties.lookupOrDefault<scalar>("A_gamma", 0.00043));
dimensionedScalar R_gas("R_gas", dimEnergy/dimTemperature/dimMoles, waamProperties.lookupOrDefault<scalar>("R_gas", 8.314));
dimensionedScalar Gamma_s("Gamma_s", dimMoles/dimArea, waamProperties.lookupOrDefault<scalar>("Gamma_s", 1.3e-5));
dimensionedScalar k1_gamma("k1_gamma", dimless, waamProperties.lookupOrDefault<scalar>("k1_gamma", 0.00318));
dimensionedScalar wt_S("wt_S", dimless, waamProperties.lookupOrDefault<scalar>("wt_S", 0.018));
dimensionedScalar delta_H0("delta_H0", dimEnergy/dimMoles, waamProperties.lookupOrDefault<scalar>("delta_H0", -166000.0));
dimensionedScalar T_m("T_m", dimTemperature, waamProperties.lookupOrDefault<scalar>("T_m", 1809.0));

// Arc Heat Source (Gaussian - Paper Calibration)
dimensionedScalar UI_arc("UI_arc", dimPower, waamProperties.lookupOrDefault<scalar>("UI_arc", 2165.0));
dimensionedScalar eta_arc("eta_arc", dimless, waamProperties.lookupOrDefault<scalar>("eta_arc", 0.68));
dimensionedScalar eta_d("eta_d", dimless, waamProperties.lookupOrDefault<scalar>("eta_d", 0.25));
dimensionedScalar sigma_arc("sigma_arc", dimLength, waamProperties.lookupOrDefault<scalar>("sigma_arc", 0.0022));
dimensionedVector v_weld("v_weld", dimVelocity, waamProperties.lookupOrDefault<vector>("v_weld", vector(0.00283, 0.0, 0.0)));
dimensionedVector x0_torch("x0_torch", dimLength, waamProperties.lookupOrDefault<vector>("x0_torch", vector(0.015, 0.0, 0.008)));

// Discrete Droplet Generation
dimensionedScalar f_d("f_d", dimless/dimTime, waamProperties.lookupOrDefault<scalar>("f_d", 60.0));
dimensionedScalar d_wire("d_wire", dimLength, waamProperties.lookupOrDefault<scalar>("d_wire", 0.0012));
dimensionedScalar v_wire("v_wire", dimVelocity, waamProperties.lookupOrDefault<scalar>("v_wire", 0.0783));
dimensionedScalar T_drop("T_drop", dimTemperature, waamProperties.lookupOrDefault<scalar>("T_drop", 1900.0));
dimensionedVector v_drop("v_drop", dimVelocity, waamProperties.lookupOrDefault<vector>("v_drop", vector(0, 0, -0.7)));
dimensionedScalar h_drop("h_drop", dimLength, waamProperties.lookupOrDefault<scalar>("h_drop", 0.0015));
"""

with open(os.path.join(WAAM_SOLVER_SRC, 'readWAAMProperties.H'), 'w') as f:
    f.write(readWAAMProperties_H)

with open(os.path.join(WAAM_SOLVER_SRC, 'waamFoam.C'), 'r') as f:
    content = f.read()

# Fix waamFoam.C syntax issue from last rewrite
content = content.replace('Info<< "Time = " << runTime.timeName() << nl << endl;\\n', 'Info<< "Time = " << runTime.timeName() << nl << endl;')

with open(os.path.join(WAAM_SOLVER_SRC, 'waamFoam.C'), 'w') as f:
    f.write(content)

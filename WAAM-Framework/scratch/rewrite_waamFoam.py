import os

# See scratch/create_0.py: research paths are environment-overridable.
WAAM_SCRATCH_ROOT = os.environ.get("WAAM_SCRATCH_ROOT", "/Users/pandeyji/Desktop/WAAM")
WAAM_SOLVER_SRC = os.path.join(WAAM_SCRATCH_ROOT, "waamFoam")
WAAM_CASE_DIR = os.path.join(WAAM_SCRATCH_ROOT, "waamCase")

waamProperties = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2412                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      waamProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Thermal & Phase Change Properties (ER70S6 Steel)
T_sol           1698.0;     // Solidus temperature [K]
T_liq           1778.0;     // Liquidus temperature [K]
L_fusion        2.7e5;      // Latent heat of fusion [J/kg]
Cp_metal        750.0;      // Specific heat capacity of metal [J/(kg K)]
Cp_gas          1000.0;     // Specific heat capacity of shielding gas [J/(kg K)]
k_metal         25.0;       // Thermal conductivity of metal [W/(m K)]
k_gas           0.05;       // Thermal conductivity of shielding gas [W/(m K)]
beta_T          1.6e-5;     // Thermal expansion coefficient [1/K]
T_ref           300.0;      // Ambient reference temperature [K]

// Hydrodynamic & Porosity properties
C_mush          1.0e7;      // Darcy drag coefficient for mushy zone [kg/(m^3 s)]

// Marangoni Surface Tension (Fe-S Pseudo-binary System)
gamma_m0        1.943;      // Surface tension of pure Fe at Tm [N/m]
A_gamma         0.00043;    // Temperature coefficient [N/(m K)]
R_gas           8.314;      // Universal gas constant [J/(mol K)]
Gamma_s         1.3e-5;     // Surface excess at saturation [mol/m^2]
k1_gamma        0.00318;    // Entropy factor [-]
wt_S            0.018;      // Weight percent of Sulfur (ER70S6)
delta_H0        -166000.0;  // Heat of adsorption [J/mol]
T_m             1809.0;     // Melting point of pure Fe [K]

// Arc Heat Source (Gaussian - Paper Calibration)
UI_arc          2165.0;     // Average arc power UI [W]
eta_arc         0.68;       // Total arc efficiency
eta_d           0.25;       // Droplet heat ratio
sigma_arc       0.0022;     // Effective radius of heat distribution [m]
v_weld          (0.00283 0 0); // Welding travel velocity (0.17 m/min = 2.83 mm/s) [m/s]
x0_torch        (0.015 0 0.008); // Initial torch position [m]

// Discrete Droplet Generation (Paper Parameters)
f_d             60.0;       // Droplet frequency [Hz]
d_wire          0.0012;     // Wire diameter (1.2 mm) [m]
v_wire          0.0783;     // Wire feed speed (4.7 m/min = 0.0783 m/s) [m/s]
T_drop          1900.0;     // Droplet temperature [K]
v_drop          (0 0 -0.7); // Initial droplet velocity [m/s]
h_drop          0.0015;     // Generation height above current surface [m]

// ************************************************************************* //
"""

updateWAAMProperties_H = """//- Update WAAM Multiphysics Fields

const scalar timeVal = runTime.value();
const vector torchPos = x0_torch.value() + v_weld.value()*timeVal;

const scalar pi = constant::mathematical::pi;

// 1. Calculate realistic Gaussian Heat Flux (applied as volumetric source near surface)
const scalar q_total = eta_arc.value() * (1.0 - eta_d.value()) * UI_arc.value();
const scalar sig2 = sqr(sigma_arc.value());
const scalar q_max = q_total / (pi * sig2);

volVectorField gradAlpha1(fvc::grad(alpha1));
volScalarField magGradAlpha1(mag(gradAlpha1));
volVectorField gradT(fvc::grad(T));

forAll(mesh.C(), celli)
{
    const vector& c = mesh.C()[celli];
    scalar dx = c.x() - torchPos.x();
    scalar dy = c.y() - torchPos.y();
    
    // Gaussian distribution
    scalar qVal = q_max * Foam::exp(-(sqr(dx) + sqr(dy)) / sig2);
    
    // Apply arc heat heavily at the free surface (interface cells)
    // magGradAlpha1 acts as a Dirac delta at the surface
    if (alpha1[celli] > 0.01 && alpha1[celli] < 0.99)
    {
        Q_arc[celli] = qVal * magGradAlpha1[celli];
    }
    else
    {
        Q_arc[celli] = 0.0;
    }
}
Q_arc.correctBoundaryConditions();

// 2. Liquid and Solid fraction updates (Enthalpy-Porosity)
forAll(mesh.C(), celli)
{
    scalar T_val = T[celli];
    if (T_val <= T_sol.value())
    {
        fl[celli] = 0.0;
    }
    else if (T_val >= T_liq.value())
    {
        fl[celli] = 1.0;
    }
    else
    {
        fl[celli] = (T_val - T_sol.value()) / (T_liq.value() - T_sol.value() + 1e-6);
    }
}
fl.correctBoundaryConditions();
fs = scalar(1.0) - fl;

// 3. Darcy mushy-zone damping
forAll(mesh.C(), celli)
{
    scalar fl_val = fl[celli];
    scalar a1_val = alpha1[celli];
    if (a1_val > 0.01)
    {
        Sp_mush[celli] = a1_val * C_mush.value() * sqr(1.0 - fl_val) / (pow3(fl_val) + 1e-3);
    }
    else
    {
        Sp_mush[celli] = 0.0;
    }
}
Sp_mush.correctBoundaryConditions();

// 4. Effective Thermophysical Properties (with Latent Heat of Fusion)
Cp_eff = alpha1*Cp_metal + alpha2*Cp_gas;
forAll(mesh.C(), celli)
{
    scalar T_val = T[celli];
    if (T_val >= T_sol.value() && T_val <= T_liq.value())
    {
        Cp_eff[celli] += alpha1[celli] * L_fusion.value() / (T_liq.value() - T_sol.value() + 1e-6);
    }
}
Cp_eff.correctBoundaryConditions();

k_eff = alpha1*k_metal + alpha2*k_gas;
k_eff.correctBoundaryConditions();

// 5. Marangoni tangential thermocapillary force (Fe-S model)
const scalar R_g = R_gas.value();
const scalar Gam_s = Gamma_s.value();
const scalar k1_a = k1_gamma.value() * wt_S.value();
const scalar dH = delta_H0.value();
const scalar A_g = A_gamma.value();

forAll(mesh.C(), celli)
{
    if (magGradAlpha1[celli] > 1e-2 && alpha1[celli] > 0.01 && alpha1[celli] < 0.99)
    {
        scalar T_val = max(T[celli], T_sol.value());
        
        scalar expTerm = Foam::exp(-dH / (R_g * T_val));
        scalar K_val = k1_a * expTerm;
        
        // Analytical derivative: d(gamma)/dT
        scalar dgam_dT = -A_g - R_g * Gam_s * Foam::log(1.0 + K_val) 
                         - Gam_s * (K_val / (1.0 + K_val)) * (dH / T_val);
                         
        vector n = gradAlpha1[celli] / (magGradAlpha1[celli] + 1e-8);
        vector gradT_tan = gradT[celli] - (gradT[celli] & n)*n;
        
        F_marangoni[celli] = dgam_dT * gradT_tan * magGradAlpha1[celli];
    }
    else
    {
        F_marangoni[celli] = vector::zero;
    }
}
F_marangoni.correctBoundaryConditions();

// 6. Thermal Buoyancy force
F_buoyancy = -rho * g * beta_T * (T - T_ref) * alpha1 * fl;
F_buoyancy.correctBoundaryConditions();

// We set VOF_source to 0 since we use explicit discrete droplets
VOF_source == 0.0;
"""

waamFoam_patch = """
    // Add logic at start of time step for Droplet generation
    static label lastDropletIndex = -1;
    const scalar fd = readScalar(waamProperties.lookup("f_d"));
    label currentDropletIndex = floor(runTime.value() * fd);
    
    if (currentDropletIndex > lastDropletIndex && currentDropletIndex > 0)
    {
        Info << "Injecting Droplet " << currentDropletIndex << " at time " << runTime.value() << " s" << endl;
        
        const scalar v_w = readScalar(waamProperties.lookup("v_wire"));
        const scalar d_w = readScalar(waamProperties.lookup("d_wire"));
        const scalar vol_rate = 0.25 * constant::mathematical::pi * d_w * d_w * v_w;
        const scalar v_drop = vol_rate / fd;
        const scalar r_drop = Foam::pow(0.75 * v_drop / constant::mathematical::pi, 1.0/3.0);
        
        const vector torchPos = vector(waamProperties.lookup("x0_torch")) 
                                + vector(waamProperties.lookup("v_weld")) * runTime.value();
        
        // Find highest z of the wall under the torch
        scalar highestZ = 0.008; // substrate height
        forAll(mesh.C(), celli)
        {
            if (alpha1[celli] > 0.5)
            {
                const vector& c = mesh.C()[celli];
                if (sqr(c.x() - torchPos.x()) + sqr(c.y() - torchPos.y()) < sqr(0.005))
                {
                    if (c.z() > highestZ)
                    {
                        highestZ = c.z();
                    }
                }
            }
        }
        
        const scalar h_d = readScalar(waamProperties.lookup("h_drop"));
        const vector dropCenter(torchPos.x(), torchPos.y(), highestZ + h_d);
        const vector dropVelocity(waamProperties.lookup("v_drop"));
        const scalar dropTemp = readScalar(waamProperties.lookup("T_drop"));
        
        scalar injectedVolume = 0.0;
        
        forAll(mesh.C(), celli)
        {
            const vector& c = mesh.C()[celli];
            if (mag(c - dropCenter) <= r_drop)
            {
                if (alpha1[celli] < 0.99)
                {
                    alpha1[celli] = 1.0;
                    T[celli] = dropTemp;
                    U[celli] = dropVelocity;
                    injectedVolume += mesh.V()[celli];
                }
            }
        }
        alpha1.correctBoundaryConditions();
        T.correctBoundaryConditions();
        U.correctBoundaryConditions();
        
        alpha2 = 1.0 - alpha1;
        mixture.correct();
        
        Info << "Injected droplet volume: " << injectedVolume << " (Ideal: " << v_drop << ")" << endl;
        lastDropletIndex = currentDropletIndex;
    }
"""

with open(os.path.join(WAAM_CASE_DIR, 'constant/waamProperties'), 'w') as f:
    f.write(waamProperties)

with open(os.path.join(WAAM_SOLVER_SRC, 'updateWAAMProperties.H'), 'w') as f:
    f.write(updateWAAMProperties_H)

# Insert the patch into waamFoam.C
with open(os.path.join(WAAM_SOLVER_SRC, 'waamFoam.C'), 'r') as f:
    content = f.read()

patch_anchor = "Info<< \"Time = \" << runTime.timeName() << nl << endl;"
new_content = content.replace(patch_anchor, patch_anchor + "\\n" + waamFoam_patch)

with open(os.path.join(WAAM_SOLVER_SRC, 'waamFoam.C'), 'w') as f:
    f.write(new_content)

print("waamFoam updated successfully with discrete droplet generation and Fe-S Marangoni physics.")

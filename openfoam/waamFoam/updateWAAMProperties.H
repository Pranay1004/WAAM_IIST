//- Update WAAM Multiphysics Fields

const scalar timeVal = runTime.value();
const scalar pi = constant::mathematical::pi;

// Multi-layer Path Scheduling
const scalar tPass = layerDuration.value();
const scalar tCool = coolingDuration.value();
const scalar tCycle = tPass + tCool;

const label currentLayer = (tCycle > 1e-6) ? static_cast<label>(timeVal / tCycle) : 0;
const scalar tInCycle = (tCycle > 1e-6) ? (timeVal - currentLayer * tCycle) : timeVal;

bool arcActive = false;
vector torchPos = x0_torch.value();

if (currentLayer < nLayers)
{
    if (tInCycle <= tPass)
    {
        arcActive = true;
        if (bidirectional && (currentLayer % 2 == 1))
        {
            // Reverse direction for odd layers: starts at (x0 + v*tPass) and moves backwards
            const vector x_end = x0_torch.value() + v_weld.value() * tPass;
            torchPos = x_end - v_weld.value() * tInCycle;
        }
        else
        {
            // Forward direction for even layers: starts at x0 and moves forwards
            torchPos = x0_torch.value() + v_weld.value() * tInCycle;
        }
    }
    else
    {
        // Interpass cooling period: arc & wire OFF, torch rests at the pass end
        arcActive = false;
        if (bidirectional && (currentLayer % 2 == 1))
        {
            torchPos = x0_torch.value();
        }
        else
        {
            torchPos = x0_torch.value() + v_weld.value() * tPass;
        }
    }
}
else
{
    // Post-weld cooling: all layers completed
    arcActive = false;
    torchPos = x0_torch.value() + v_weld.value() * tPass;
}

static label lastReportedLayer = -1;
static bool lastReportedActive = false;
if (currentLayer != lastReportedLayer || arcActive != lastReportedActive)
{
    if (arcActive)
    {
        Info<< "--> WAAM Path: LAYER " << (currentLayer + 1) << " / " << nLayers
            << " ACTIVE [t = " << timeVal << " s, torchPos = " << torchPos << "]" << endl;
    }
    else if (currentLayer < nLayers)
    {
        Info<< "--> WAAM Path: INTERPASS COOLING for Layer " << (currentLayer + 1)
            << " [t = " << timeVal << " s, dwell = " << tCool << " s]" << endl;
    }
    else
    {
        Info<< "--> WAAM Path: ALL " << nLayers << " LAYERS COMPLETE - Post-weld cooling active [t = "
            << timeVal << " s]" << endl;
    }
    lastReportedLayer = currentLayer;
    lastReportedActive = arcActive;
}

volVectorField gradAlpha1(fvc::grad(alpha1));
volScalarField magGradAlpha1(mag(gradAlpha1));
volVectorField gradT(fvc::grad(T));

// 1. Find the highest surface of deposited metal under the torch
scalar highestZ = 0.008; // substrate surface
forAll(mesh.C(), celli)
{
    if (alpha1[celli] > 0.5)
    {
        const vector& c = mesh.C()[celli];
        scalar dxy2 = sqr(c.x() - torchPos.x()) + sqr(c.y() - torchPos.y());
        if (dxy2 < sqr(0.003)) // within 3 mm of torch
        {
            if (c.z() > highestZ)
            {
                highestZ = c.z();
            }
        }
    }
}
reduce(highestZ, maxOp<scalar>());

// 2. Wire Mass Deposition (VOF Source): 100% Volume-Conserved Gaussian Kernel
const scalar d_w = d_wire.value();
const scalar v_w = arcActive ? v_wire.value() : 0.0;
const scalar vol_rate = 0.25 * pi * d_w * d_w * v_w; // m^3/s


const vector depCenter(torchPos.x(), torchPos.y(), highestZ + 0.0003);
const scalar sig_xy2 = sqr(0.0015);
const scalar sig_z2 = sqr(0.0006);

scalar totalWeightV = 0.0;
forAll(mesh.C(), celli)
{
    const vector& c = mesh.C()[celli];
    scalar dx2 = sqr(c.x() - depCenter.x());
    scalar dy2 = sqr(c.y() - depCenter.y());
    scalar dz2 = sqr(c.z() - depCenter.z());
    
    // Deposit into non-full cells at or above the substrate surface
    if (c.z() >= (highestZ - 0.0005) && alpha1[celli] < 0.99)
    {
        scalar w = Foam::exp(- (dx2 + dy2)/(2.0*sig_xy2) - dz2/(2.0*sig_z2)) * (1.0 - alpha1[celli]);
        totalWeightV += w * mesh.V()[celli];
    }
}
reduce(totalWeightV, sumOp<scalar>());

if (totalWeightV > 1e-15)
{
    const scalar rate_factor = vol_rate / totalWeightV;
    forAll(mesh.C(), celli)
    {
        const vector& c = mesh.C()[celli];
        scalar dx2 = sqr(c.x() - depCenter.x());
        scalar dy2 = sqr(c.y() - depCenter.y());
        scalar dz2 = sqr(c.z() - depCenter.z());
        
        if (c.z() >= (highestZ - 0.0005) && alpha1[celli] < 0.99)
        {
            scalar w = Foam::exp(- (dx2 + dy2)/(2.0*sig_xy2) - dz2/(2.0*sig_z2)) * (1.0 - alpha1[celli]);
            VOF_source[celli] = rate_factor * w;
        }
        else
        {
            VOF_source[celli] = 0.0;
        }
    }
}
else
{
    VOF_source == dimensionedScalar("zero", dimless/dimTime, 0.0);
}
VOF_source.correctBoundaryConditions();

// Direct volume addition is not needed, handled natively by MULES via alphaSuSp.H

// 3. Arc Heat Flux: 100% Energy-Conserved Surface Source
const scalar q_total = arcActive ? (eta_arc.value() * (1.0 - eta_d.value()) * UI_arc.value()) : 0.0;
const scalar sig_arc2 = sqr(sigma_arc.value());

scalar totalWeightQ = 0.0;
forAll(mesh.C(), celli)
{
    const vector& c = mesh.C()[celli];
    scalar dx2 = sqr(c.x() - torchPos.x());
    scalar dy2 = sqr(c.y() - torchPos.y());
    
    if (alpha1[celli] > 0.05 && c.z() >= (highestZ - 0.001))
    {
        scalar w_q = Foam::exp(-(dx2 + dy2) / sig_arc2) * (magGradAlpha1[celli] + 100.0 * alpha1[celli]);
        totalWeightQ += w_q * mesh.V()[celli];
    }
}
reduce(totalWeightQ, sumOp<scalar>());

if (totalWeightQ > 1e-15)
{
    const scalar q_factor = q_total / totalWeightQ;
    forAll(mesh.C(), celli)
    {
        const vector& c = mesh.C()[celli];
        scalar dx2 = sqr(c.x() - torchPos.x());
        scalar dy2 = sqr(c.y() - torchPos.y());
        
        if (alpha1[celli] > 0.05 && c.z() >= (highestZ - 0.001))
        {
            scalar w_q = Foam::exp(-(dx2 + dy2) / sig_arc2) * (magGradAlpha1[celli] + 100.0 * alpha1[celli]);
            Q_arc[celli] = q_factor * w_q;
        }
        else
        {
            Q_arc[celli] = 0.0;
        }
    }
}
else
{
    Q_arc == dimensionedScalar("zero", dimPower/dimVolume, 0.0);
}
Q_arc.correctBoundaryConditions();

// 4. Liquid and Solid fraction updates (Enthalpy-Porosity)
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

// 5. Darcy mushy-zone damping
forAll(mesh.C(), celli)
{
    scalar fl_val = fl[celli];
    scalar a1_val = alpha1[celli];
    if (a1_val > 0.05)
    {
        Sp_mush[celli] = a1_val * C_mush.value() * sqr(1.0 - fl_val) / (pow3(fl_val) + 1e-3);
    }
    else
    {
        Sp_mush[celli] = 0.0;
    }
}
Sp_mush.correctBoundaryConditions();

// 6. Effective Thermophysical Properties (with Latent Heat of Fusion)
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

// 7. Marangoni tangential thermocapillary force (Fe-S model)
const scalar R_g = R_gas.value();
const scalar Gam_s = Gamma_s.value();
const scalar k1_a = k1_gamma.value() * wt_S.value();
const scalar dH = delta_H0.value();
const scalar A_g = A_gamma.value();

forAll(mesh.C(), celli)
{
    if (magGradAlpha1[celli] > 1e-2 && alpha1[celli] > 0.05 && alpha1[celli] < 0.95 && T[celli] >= T_sol.value())
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

// 8. Thermal Buoyancy force
F_buoyancy = -rho * g * beta_T * (T - T_ref) * alpha1 * fl;
F_buoyancy.correctBoundaryConditions();

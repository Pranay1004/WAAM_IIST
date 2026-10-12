{
    surfaceScalarField rhoCpPhi
    (
        "rhoCpPhi",
        fvc::interpolate(rho*Cp_eff)*phi
    );

    fvScalarMatrix TEqn
    (
        fvm::ddt(rho*Cp_eff, T)
      + fvm::div(rhoCpPhi, T)
      - fvm::laplacian(k_eff, T)
     ==
        Q_arc
      + VOF_source*rho1*Cp_metal*T_drop
    );

    TEqn.relax();
    TEqn.solve();

    // Physical temperature bounding
    T.max(dimensionedScalar("T_min", dimTemperature, 290.0));
    T.min(dimensionedScalar("T_max", dimTemperature, 3500.0));
    T.correctBoundaryConditions();
}

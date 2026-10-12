#include "readWAAMProperties.H"

Info<< "Creating field T (Temperature)\n" << endl;
volScalarField T
(
    IOobject
    (
        "T",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Creating field fl (Liquid fraction)\n" << endl;
volScalarField fl
(
    IOobject
    (
        "fl",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("fl", dimless, 0.0)
);

Info<< "Creating field fs (Solid fraction)\n" << endl;
volScalarField fs
(
    IOobject
    (
        "fs",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    scalar(1.0) - fl
);

Info<< "Creating field Q_arc (Arc heat source)\n" << endl;
volScalarField Q_arc
(
    IOobject
    (
        "Q_arc",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("Q_arc", dimPower/dimVolume, 0.0)
);

Info<< "Creating field Sp_mush (Darcy mushy zone damping)\n" << endl;
volScalarField Sp_mush
(
    IOobject
    (
        "Sp_mush",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("Sp_mush", dimDensity/dimTime, 0.0)
);

Info<< "Creating field VOF_source (Wire mass deposition rate)\n" << endl;
volScalarField VOF_source
(
    IOobject
    (
        "VOF_source",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("VOF_source", dimless/dimTime, 0.0)
);

Info<< "Creating field F_marangoni (Marangoni surface force)\n" << endl;
volVectorField F_marangoni
(
    IOobject
    (
        "F_marangoni",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("F_marangoni", dimForce/dimVolume, vector::zero)
);

Info<< "Creating field F_buoyancy (Thermal buoyancy force)\n" << endl;
volVectorField F_buoyancy
(
    IOobject
    (
        "F_buoyancy",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("F_buoyancy", dimForce/dimVolume, vector::zero)
);

Info<< "Creating thermophysical property fields\n" << endl;
volScalarField Cp_eff
(
    IOobject
    (
        "Cp_eff",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    alpha1*Cp_metal + alpha2*Cp_gas
);

volScalarField k_eff
(
    IOobject
    (
        "k_eff",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    alpha1*k_metal + alpha2*k_gas
);

mesh.setFluxRequired(T.name());

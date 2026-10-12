/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2412                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      alpha.metal;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    bottom
    {
        type            zeroGradient;
        value           uniform 0;
    }
    top
    {
        type            inletOutlet;
        value           uniform 0;
        inletValue      uniform 0;
    }
    inlet
    {
        type            zeroGradient;
        value           uniform 0;
    }
    outlet
    {
        type            zeroGradient;
        value           uniform 0;
    }
    sides
    {
        type            zeroGradient;
        value           uniform 0;
    }
}

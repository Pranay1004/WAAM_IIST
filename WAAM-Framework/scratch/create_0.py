import os

import sys as _sys

# Research scratch script: paths come from the environment so the file works
# outside the machine it was written on. Previously every path below was a
# hardcoded /Users/pandeyji/... absolute path (and one os.system("cp -r ...")
# with interpolated paths), so the script could only ever run -- destructively
# -- on its author's laptop.
WAAM_SCRATCH_ROOT = os.environ.get(
    "WAAM_SCRATCH_ROOT", "/Users/pandeyji/Desktop/WAAM/waamCase")
zero_dir = os.path.join(WAAM_SCRATCH_ROOT, "0")
os.makedirs(zero_dir, exist_ok=True)

header = """/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       vol%sField;
    object      %s;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      %s;

internalField   uniform %s;

boundaryField
{
    bottom
    {
        type            %s;
        value           uniform %s;
    }
    top
    {
        type            %s;
        value           uniform %s;
%s    }
    inlet
    {
        type            %s;
        value           uniform %s;
%s    }
    outlet
    {
        type            %s;
        value           uniform %s;
%s    }
    sides
    {
        type            %s;
        value           uniform %s;
    }
}
"""

def write_field(name, field_type, dims, internal, b_type_bot, b_val_bot, b_type_top, b_val_top, b_type_sides, b_val_sides):
    inlet_str_top = ""
    if b_type_top == "inletOutlet":
        inlet_str_top += f"        inletValue      uniform {b_val_top};\n"
    if b_type_top == "totalPressure":
        inlet_str_top += f"        p0      uniform {b_val_top};\n"
        
    inlet_str_inlet = ""
    if b_type_sides == "inletOutlet":
        inlet_str_inlet += f"        inletValue      uniform {b_val_sides};\n"
    if b_type_sides == "totalPressure":
        inlet_str_inlet += f"        p0      uniform {b_val_sides};\n"
        
    inlet_str_outlet = ""
    if b_type_sides == "inletOutlet":
        inlet_str_outlet += f"        inletValue      uniform {b_val_sides};\n"
    if b_type_sides == "totalPressure":
        inlet_str_outlet += f"        p0      uniform {b_val_sides};\n"
    
    content = header % (field_type, name, dims, internal, 
                        b_type_bot, b_val_bot, 
                        b_type_top, b_val_top, inlet_str_top,
                        b_type_sides, b_val_sides, inlet_str_inlet,
                        b_type_sides, b_val_sides, inlet_str_outlet,
                        b_type_sides, b_val_sides)
    with open(os.path.join(zero_dir, name), 'w') as f:
        f.write(content)

write_field("U", "Vector", "[0 1 -1 0 0 0 0]", "(0 0 0)", "noSlip", "(0 0 0)", "pressureInletOutletVelocity", "(0 0 0)", "zeroGradient", "(0 0 0)")
write_field("T", "Scalar", "[0 0 0 1 0 0 0]", "300", "fixedValue", "300", "inletOutlet", "300", "zeroGradient", "300")
write_field("p_rgh", "Scalar", "[1 -1 -2 0 0 0 0]", "100000", "fixedFluxPressure", "100000", "totalPressure", "100000", "zeroGradient", "100000")
write_field("alpha.metal", "Scalar", "[0 0 0 0 0 0 0]", "0", "zeroGradient", "0", "inletOutlet", "0", "zeroGradient", "0")
write_field("alpha.gas", "Scalar", "[0 0 0 0 0 0 0]", "1", "zeroGradient", "1", "inletOutlet", "1", "zeroGradient", "1")

import shutil as _shutil
_shutil.copytree(zero_dir, os.path.join(WAAM_SCRATCH_ROOT, "0.orig"),
                 dirs_exist_ok=True)

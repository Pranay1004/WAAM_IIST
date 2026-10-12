import os
import subprocess
import time
import re
import shutil

cores_to_test = [1, 2, 3, 4, 6]
bench_endTime = 0.003

openfoam_env = "/opt/homebrew/bin/openfoam2412 -c"
solver = "/Users/pandeyji/OpenFOAM/pandeyji-v2412/platforms/darwin64ClangDPInt32Opt/bin/waamFoam"

# Ensure controlDict endTime is bench_endTime
with open("system/controlDict", "r") as f:
    control_content = f.read()

control_bench = re.sub(r"endTime\s+[\d\.]+;", f"endTime         {bench_endTime};", control_content)
with open("system/controlDict", "w") as f:
    f.write(control_bench)

results = {}

def clean_case():
    for item in os.listdir("."):
        if item.startswith("processor") or item == "VTK" or item == "0":
            if os.path.isdir(item):
                shutil.rmtree(item, ignore_errors=True)
        elif re.match(r"^[0-9]+(\.[0-9]+)?$", item):
            if os.path.isdir(item):
                shutil.rmtree(item, ignore_errors=True)
    if os.path.exists("0"):
        shutil.rmtree("0", ignore_errors=True)
    shutil.copytree("0.orig", "0")

for cores in cores_to_test:
    print(f"\n==========================================")
    print(f"BENCHMARKING: {cores} CORE(S)")
    print(f"==========================================")
    
    clean_case()
    
    if cores > 1:
        decomp_text = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2412                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains {cores};

method          scotch;

// ************************************************************************* //
"""
        with open("system/decomposeParDict", "w") as f:
            f.write(decomp_text)
            
        print(f"Decomposing mesh for {cores} domains...")
        p_dec = subprocess.run(f'{openfoam_env} "decomposePar -force"', shell=True, capture_output=True, text=True)
        if p_dec.returncode != 0:
            print(f"Decompose failed! Error:\n{p_dec.stderr}\n{p_dec.stdout}")
        
    log_file = f"log.waamFoam_{cores}"
    print(f"Running waamFoam on {cores} core(s)...")
    
    wall_start = time.time()
    if cores == 1:
        cmd = f'{openfoam_env} "{solver} > {log_file} 2>&1"'
    else:
        cmd = f'{openfoam_env} "mpirun -np {cores} {solver} -parallel > {log_file} 2>&1"'
        
    subprocess.run(cmd, shell=True)
    wall_duration = time.time() - wall_start
    
    exec_time = None
    clock_time = None
    steps = 0
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Time = " in line and not "ExecutionTime" in line:
                    steps += 1
            for line in reversed(lines):
                if "ExecutionTime =" in line:
                    m = re.search(r"ExecutionTime\s*=\s*([\d\.]+)\s*s\s+ClockTime\s*=\s*([\d\.]+)\s*s", line)
                    if m:
                        exec_time = float(m.group(1))
                        clock_time = float(m.group(2))
                    break
    except Exception as e:
        print(f"Error reading log: {e}")
        
    results[cores] = {
        "exec_time": exec_time,
        "clock_time": clock_time if clock_time is not None else wall_duration,
        "wall_time": wall_duration,
        "steps": steps
    }
    print(f"Result for {cores} core(s): ExecutionTime={exec_time}s, ClockTime={clock_time}s, WallDuration={wall_duration:.2f}s")

# Restore controlDict
with open("system/controlDict", "w") as f:
    f.write(control_content)

# Restore decomposeParDict to 4 cores scotch
decomp_text = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2412                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains 4;

method          scotch;

// ************************************************************************* //
"""
with open("system/decomposeParDict", "w") as f:
    f.write(decomp_text)

print("\n" + "="*65)
print(f"{'CORES':<8} | {'CLOCK TIME (s)':<15} | {'EXEC TIME (s)':<15} | {'SPEEDUP':<10} | {'EFFICIENCY':<10}")
print("="*65)
base_time = results[1]["clock_time"] if results[1]["clock_time"] else 1.0

for c in cores_to_test:
    ct = results[c]["clock_time"]
    et = results[c]["exec_time"]
    speedup = base_time / ct if ct and ct > 0 else 0.0
    efficiency = (speedup / c) * 100.0 if c > 0 else 0.0
    et_str = f"{et:.2f}" if et is not None else "N/A"
    ct_str = f"{ct:.2f}" if ct is not None else "N/A"
    print(f"{c:<8} | {ct_str:<15} | {et_str:<15} | {speedup:<9.2f}x | {efficiency:<9.1f}%")

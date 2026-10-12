import os
import shutil

case_dir = "/Users/pandeyji/Desktop/WAAM/waamCase"
keep_times = {0.0} # always keep 0 and 0.orig
keep_times_str = {"0", "0.orig", "constant", "system", "VTK"}

# Generate times to keep: 0.1, 0.5, 1.0, 1.5, 2.0, ... 10.0
for i in range(1, 100):
    val = round(i * 0.1, 1)
    if val % 0.5 == 0.0 or val == 0.1: # keep 0.1, 0.5, 1.0, 1.5...
        keep_times.add(val)
        if val.is_integer():
            keep_times_str.add(str(int(val)))
        else:
            keep_times_str.add(str(val))

print(f"Keeping exact matches: {keep_times_str}")

# Also keep the very last time
time_dirs = []
for entry in os.listdir(case_dir):
    path = os.path.join(case_dir, entry)
    if os.path.isdir(path):
        try:
            t = float(entry)
            time_dirs.append((t, entry))
        except ValueError:
            pass

if time_dirs:
    time_dirs.sort()
    latest_dir = time_dirs[-1][1]
    keep_times_str.add(latest_dir)
    print(f"Keeping latest time: {latest_dir}")

for entry in os.listdir(case_dir):
    path = os.path.join(case_dir, entry)
    if os.path.isdir(path):
        # Is it a numeric time dir?
        try:
            t = float(entry)
            if entry not in keep_times_str:
                print(f"Deleting {entry}")
                shutil.rmtree(path)
        except ValueError:
            pass

# Also clean processor directories
for entry in os.listdir(case_dir):
    if entry.startswith("processor"):
        path = os.path.join(case_dir, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Deleted {entry}")

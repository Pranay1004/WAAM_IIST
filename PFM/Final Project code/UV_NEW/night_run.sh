#!/bin/bash
# UV_NEW overnight queue — sequential, one job at a time (each uses all cores in stencil phase)
# Estimated total ~5 h. Log: night.log
R=/Users/pandeyji/Desktop/WAAM/PFM/Final\ Project\ code/UV_NEW/results
run() {
  echo "===== START $1 $(date) =====" >> "$R/night.log"
  ./run_"$1" >> stdout.txt 2>&1
  echo "===== END $1 exit=$? $(date) =====" >> "$R/night.log"
}
cd "$R/binary_nondm" 2>/dev/null || true
cd "/Users/pandeyji/Desktop/WAAM/PFM/Final Project code/UV_NEW/results/bn_ctl" && run bn_ctl run_bn_ctl
cd "../bn_mild" && run bn_mild run_bn_mild
cd "../bn_strong" && run bn_strong run_bn_strong
cd "../tip_smild" && run tip_smild run_tip_smild
cd "../tip_sstrong" && run tip_sstrong run_tip_sstrong
cd "../arm_nctl" && run arm_nctl run_arm_nctl
cd "../arm_smild" && run arm_smild run_arm_smild
cd "../arm_sstrong" && run arm_sstrong run_arm_sstrong
echo "ALL NIGHT RUNS DONE $(date)" >> "/Users/pandeyji/Desktop/WAAM/PFM/Final Project code/UV_NEW/results/night.log"

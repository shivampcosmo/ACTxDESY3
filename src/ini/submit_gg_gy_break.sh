#!/bin/bash -l
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --constraint=haswell
#SBATCH --tasks-per-node=32
#SBATCH --time=16:00:00
#SBATCH --time-min=11:00:00
#SBATCH --license=SCRATCH
#SBATCH --output=OUT_log_%j.txt
#SBATCH --error=ERR_log_%j.txt

source /global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/config/setup-cosmosis-nersc 
cd /global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/ini
srun cosmosis --mpi MICE_gg_gy_break.ini

#!/bin/bash -l
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --constraint=haswell
#SBATCH --tasks-per-node=32
#SBATCH --time=16:00:00
#SBATCH --license=SCRATCH
#SBATCH --output=/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/submit_files/OUT_log_P0A_P0z_P0m_alphigh_%j.txt
#SBATCH --error=/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/submit_files/ERR_log_P0A_P0z_P0m_alphigh_%j.txt

source /global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/config/setup-cosmosis-nersc 

cd /global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/ini/final_runs/

srun cosmosis --mpi sz_gty_xip_xim_HM_delz_m_IA_P0A_P0z_P0m_alphigh_highbpl_al1_PLcosmo.ini

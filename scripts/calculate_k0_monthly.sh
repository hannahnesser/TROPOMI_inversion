#!/bin/bash

#SBATCH -J save_k0
#SBATCH -o %x_%j_%a.out
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 45000
#SBATCH -t 0-01:00
#SBATCH --mail-type=END

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
YEAR="2019"
MONTH="${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS=6

## -------------------------------------------------------------------------##
## Load the environment
## -------------------------------------------------------------------------##
echo "Activating python environment"

module load Anaconda3/5.0.1-fasrc01
source activate ~/python/miniconda/envs/TROPOMI_inversion

echo "Activated ${CONDA_PREFIX}"

## -------------------------------------------------------------------------##
## Run the script
## -------------------------------------------------------------------------##
echo "Initiating script"
python -u generate_k0.py $MONTH

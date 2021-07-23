#!/bin/bash

#SBATCH -J save_k0
#SBATCH -o %x_%j_%a.out
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -p huce_cascade
#SBATCH --mem 45000
#SBATCH -t 0-24:00
#SBATCH --mail-type=END

# 30 GB should be big enough for most months. Adaptive
# memory based on the number of observations in a month might
# be a good idea. Oh well.

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
MEMORY_GB=45
# export OMP_NUM_THREADS=4

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

python_dir=$(dirname `pwd`)
python -u ${python_dir}/python/generate_k0.py $MEMORY_GB

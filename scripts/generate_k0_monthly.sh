#!/bin/bash

#SBATCH -J save_k0_monthly
#SBATCH -o %x_%j.out
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 35000
#SBATCH -t 0-02:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hnesser@g.harvard.edu

# 30 GB should be big enough for most months. Adaptive
# memory based on the number of observations in a month might
# be a good idea. Oh well.

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
DATA_DIR="${1}"
OUTPUT_DIR="${2}"
CODE_DIR="${3}"
MEMORY_GB=45

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
python -u ${python_dir}/python/generate_k0_monthly.py ${MEMORY_GB} ${DATA_DIR} ${OUTPUT_DIR} ${CODE_DIR}

## -------------------------------------------------------------------------##
## Clean up
## -------------------------------------------------------------------------##
rm -rf ${OUTPUT_DIR}dask-worker-space

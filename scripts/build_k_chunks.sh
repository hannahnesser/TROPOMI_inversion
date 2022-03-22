#!/bin/bash

#SBATCH -J build_kpi
#SBATCH -o %x_%j_%a.out
#SBATCH -c 11
#SBATCH -N 1
#SBATCH -p huce_cascade
#SBATCH --mem 40000
#SBATCH -t 0-03:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hnesser@g.harvard.edu

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
CHUNK="${SLURM_ARRAY_TASK_ID}"
DATA_DIR=${6}

## -------------------------------------------------------------------------##
## Load and prepare the environment
## -------------------------------------------------------------------------##
echo "Activating python environment"

module load Anaconda3/5.0.1-fasrc01
source activate ~/python/miniconda/envs/TROPOMI_inversion

echo "Activated ${CONDA_PREFIX}"

rm -rf ${DATA_DIR}/dask-worker-space-${CHUNK}

## -------------------------------------------------------------------------##
## Run the script
## -------------------------------------------------------------------------##
echo "Initiating script"

python_dir=$(dirname `pwd`)
python -u ${python_dir}/python/build_k.py ${CHUNK} ${@}

## -------------------------------------------------------------------------##
## Clean up
## -------------------------------------------------------------------------##
rm -rf ${DATA_DIR}/dask-worker-space-${CHUNK}

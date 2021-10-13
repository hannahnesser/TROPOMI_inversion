#!/bin/bash

#SBATCH -J build_kpi
#SBATCH -o %x_%j_%a.out
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 45000
#SBATCH -t 0-04:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hnesser@g.harvard.edu

##12
##45000
##0-04:00
## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
MONTH="${SLURM_ARRAY_TASK_ID}"

## -------------------------------------------------------------------------##
## Load and prepare the environment
## -------------------------------------------------------------------------##
echo "Activating python environment"

module load Anaconda3/5.0.1-fasrc01
source activate ~/python/miniconda/envs/TROPOMI_inversion

echo "Activated ${CONDA_PREFIX}"

rm -rf ${DATA_DIR}dask-worker-space-${MONTH}

## -------------------------------------------------------------------------##
## Run the script
## -------------------------------------------------------------------------##
echo "Initiating script"

python_dir=$(dirname `pwd`)
python -u ${python_dir}/python/build_k.py ${MONTH} ${@}

## -------------------------------------------------------------------------##
## Clean up
## -------------------------------------------------------------------------##
rm -rf ${DATA_DIR}dask-worker-space-${MONTH}

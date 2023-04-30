#!/bin/bash

#SBATCH -J save_pph
#SBATCH -o %x_%j_%a.out
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --contiguous
#SBATCH -p huce_cascade
#SBATCH --mem 45000
#SBATCH -t 0-01:00
#SBATCH --mail-type=END
#SBATCH --mail-user=hnesser@g.harvard.edu

## Eventually change to 12 cores, 45000 mem, and at least one hour. Just debugging
## for now

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
CHUNK="${SLURM_ARRAY_TASK_ID}"
SUFFIX=${11}
DATA_DIR=${3}
OPT_BC=${4}
if [[ ${OPT_BC} == "True" ]]; then
  SUFFIX="_bc${SUFFIX}"
fi

## -------------------------------------------------------------------------##
## Load and prepare the environment
## -------------------------------------------------------------------------##
echo "Activating python environment"

module load Anaconda3/5.0.1-fasrc01
source activate ~/python/miniconda/envs/TROPOMI_inversion

echo "Activated ${CONDA_PREFIX}"

rm -rf ${DATA_DIR}/pph_dask_worker${SUFFIX}_${CHUNK}/

## -------------------------------------------------------------------------##
## Run the script
## -------------------------------------------------------------------------##
echo "Initiating script"

python_dir=$(dirname `pwd`)
python -u ${python_dir}/python/generate_pph.py ${CHUNK} ${@}

## -------------------------------------------------------------------------##
## Clean up
## -------------------------------------------------------------------------##
rm -rf ${DATA_DIR}/pph_dask_worker${SUFFIX}_${CHUNK}/

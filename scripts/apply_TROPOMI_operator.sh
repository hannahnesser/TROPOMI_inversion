#!/bin/bash

#SBATCH -J TROPOMI_operator
#SBATCH -o %x_%j_%a.out
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 8000
#SBATCH -t 0-00:40
#SBATCH --mail-type=END

## -------------------------------------------------------------------------##
## Set user preferences
## -------------------------------------------------------------------------##
# Directories
TROPOMI_DATA_DIR="${1}"
BASE_DIR="${2}"
INPUT_DIR="${BASE_DIR}OutputDir/"
OUTPUT_DIR="${BASE_DIR}ProcessedDir/"

# time range
MONTH="${SLURM_ARRAY_TASK_ID}"

## -------------------------------------------------------------------------##
## Print out user preferences
## -------------------------------------------------------------------------##
echo "======================================================================="
echo "TROPOMI DATA DIRECTORY:    ${TROPOMI_DATA_DIR}"
echo "GEOS-CHEM DATA DIRECTORY:  ${INPUT_DIR}"
echo "OUTPUT DATA DIRECTORY:     ${OUTPUT_DIR}"
echo "MONTH:                     ${MONTH}"
echo "======================================================================="

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
python_dir=$(dirname `pwd`)
cd $INPUT_DIR
mkdir -p $OUTPUT_DIR

echo "Initiating script"
python -u ${python_dir}/python/TROPOMI_operator.py $TROPOMI_DATA_DIR $INPUT_DIR $OUTPUT_DIR $MONTH

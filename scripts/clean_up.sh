#!/bin/bash

#SBATCH -J clean_up
#SBATCH -o %x_%j_%a.out
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p huce_intel
#SBATCH --mem 2000
#SBATCH -t 0-01:00
#SBATCH --mail-type=END

GC_CH4_DIR=${1}

preprocess_check_1=$(ls ${GC_CH4_DIR}/*_orig | wc -w)
preprocess_check_2=$(ls ${GC_CH4_DIR}/GEOSChem.SpeciesConc*.nc4 | wc -w)
if [[ $preprocess_check_1 == 14 && $preprocess_check_2 == 366 ]]
then
  echo "Cleaning up!"
  rm HEMCO_restart.*
  rm ${GC_CH4_DIR}/*
fi

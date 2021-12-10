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
OUTPUT_DIR=${2}

# preprocess_check_1=$(ls ${GC_CH4_DIR}/*_orig | wc -w)
# preprocess_check_2=$(ls ${GC_CH4_DIR}/GEOSChem.SpeciesConc*.nc4 | wc -w)
# check_1=$(ls ${GC_CH4_DIR}/*_orig | wc -w)

# Check for post-processing
# Check thata 14 files are replaced
check_1a=$(grep 'Replacing data on' TROPOMI_operator_*.out | wc -l)

# Check that there are no nan values in
check_1b=$(grep -q 'NAN VALUES ARE PRESENT' TROPOMI_operator_*.out)

# Check that GEOS-Chem output all the necessary output
check_2=$(ls ${GC_CH4_DIR}/GEOSChem.SpeciesConc*.nc4 | wc -w)

# Check that the post-processing output worked
check_3=$(ls ${OUTPUT_DIR}/ | wc -w)

# If those criteria are met
if [[ $check_1a == 14 && $check_1b && $check_2 == 366 && $check_3 == 365 ]]
then
  echo "Cleaning up!"
  rm HEMCO_restart.*
  rm ${GC_CH4_DIR}/*
fi

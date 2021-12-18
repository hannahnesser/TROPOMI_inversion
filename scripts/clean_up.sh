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

## Check that GEOS-Chem output all the necessary output (by count)
[[ $(ls OutputDir/GEOSChem.SpeciesConc*.nc4 | wc -w) == 366 ]] && check_gc=true || check_gc=false

# Check for post-processing
## Check that stratospheric data in 14 files are replaced
[[ $(grep --no-filename 'Replacing data on' TROPOMI_operator_*.out | sort -u | wc -l) == 14 ]] && check_pp_strat=true || check_pp_strat=false

## Check that there are no nan values in the output of the operator script
$(grep -q 'NAN VALUES ARE PRESENT' TROPOMI_operator_*.out) && check_pp_nans=false || check_pp_nans=true

## Check that the post-processing output the correct number of files
[[ $(ls ProcessedDir/ | wc -w) == 365 ]] && check_pp_count=true || check_pp_count=false

# Check that the post-proocessing output is not size 0
min_file_size=($(ls -lSh ProcessedDir | tail -n 1))
min_file_size=${min_file_size[4]}
[[ $min_file_size  == 208 ]] && check_pp_size=true || check_pp_size=false

## Concatenate the post-processing checks
[[ $check_pp_strat && $check_pp_nans && $check_pp_count && $check_pp_size ]] && check_pp=true || check_pp=false

# If those criteria are met
if [[ $check_gc && $check_pp ]]
then
  echo "Cleaning up!"
  rm HEMCO_restart.*
  rm ${GC_CH4_DIR}/*
elif [[ ! $check_gc ]]
  echo "GEOS-Chem check failed."
elif [[ ! $check_pp ]]
  if [[ ! $check_pp_strat ]]
    echo "Stratospheric data replacement failed."
  elif [[ ! $check_pp_nans ]]
    echo "NaN values are present in the TROPOMI operator output."
  elif [[ ! $check_pp_count ]]
    echo "There are an incorrect number of post-processed files."
  elif [[ ! $check_pp_size ]]
    echo "There are post-processed files with size 0."
  fi
fi

exit 0

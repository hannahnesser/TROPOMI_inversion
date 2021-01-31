#!/bin/bash

# Edit apply_TROPOMI_operator.sh to change settings.

#sbatch --array=5-12 apply_TROPOMI_operator.sh

months=( 5 7 8 9 11 12 )
#months=( 4 )

for m in "${months[@]}"
do
    sbatch --export=ALL,MONTH=${m} apply_TROPOMI_operator.sh
done

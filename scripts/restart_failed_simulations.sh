cd /n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs

for x in $(seq 17 33);
do

if [ $x -lt 10 ]; then
    xstr="000${x}"
elif [ $x -lt 100 ]; then
    xstr="00${x}"
elif [ $x -lt 1000 ]; then
    xstr="0${x}"
else
    xstr="${x}"
fi

run_name="TROPOMI_inversion_${xstr}"

# Check for GC completion
GC_check=$(tail -n 2 ${run_name}/${run_name}.log | head -n 1)
if [[ $GC_check != "**************   E N D   O F   G E O S -- C H E M   **************" ]]
then
    echo "GEOS-Chem failed -- ${run_name}"
    # All GC runs worked in the last set of perturbations, so we're
    # skipping this part for now
else
    # echo "All GEOS-Chem simulations completed."
    # Check for completion of preprocessing
    preprocess_check=$(ls ${run_name}/OutputDir/*_orig | wc -w)
    if [[ $preprocess_check != 14 ]]; then
        echo "Preprocessing failed -- ${run_name}"

        # Enter the run directory
        cd ${run_name}

        # Modify the run file
        sed -i -e "s@export OMP_NUM_THREADS@# export OMP_NUM_THREADS@g" \
               -e "s@. ~/init/init@# . ~/init/init@g" \
               -e "s@./geos@# ./geos@g" ${run_name}.run
               # -e "s@##SBATCH@#SBATCH@g" \
               # -e "s@-c 16@-c 1@g" \
               # -e "s@-t 4-00:00@-t 0-00:10@g" \
               # -e "s@--mem=20000@--mem=2000@g" \


        # Remove old output file and resubmit
        rm preprocess_GC*.out
        ./${run_name}.run

        # Exit directory
        cd ..
    else
        # echo "All pre-processing completed."
        # Check for completion of operator
        operator_check=$(ls ${run_name}/ProcessedDir/ | wc -w)
        if [[ $operator_check != 365 ]]; then
            echo "Operator failed -- ${run_name}"

            # Enter the run directory
            cd ${run_name}

            # Modify the run file
            sed -i -e "s@export OMP_NUM_THREADS@# export OMP_NUM_THREADS@g" \
                   -e "s@. ~/init/init@# . ~/init/init@g" \
                   -e "s@./geos@# ./geos@g" \
                   -e "s@jid1=@# jid1=@g" \
                   -e "s@--dependency=afterok:\${jid1##\* } @@g" ${run_name}.run

            # Remove old output files and resubmit
            rm -f TROPOMI_operator_*
            ./${run_name}.run

            # Exit directory
            cd ..
        fi
    fi
fi

done

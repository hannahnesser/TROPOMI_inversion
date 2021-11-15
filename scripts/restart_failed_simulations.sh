# Boolean options
RestartProcesses=false

# Directories
jac_dir="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs"
inv_dir="/n/home04/hnesser/TROPOMI_inversion"

# Set up
cd $jac_dir

for x in $(seq 113 434);
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
run_dir="${jac_dir}/${run_name}"

# Check for GC completion
if ! grep -q 'E N D   O F   G E O S -- C H E M' "${run_dir}/${run_name}.log"
then
    echo "GEOS-Chem failed -- ${run_name}"
    # All GC runs worked in the last set of perturbations, so we're
    # skipping this part for now
else
    # Check that the output dir has been cleared
    cleanup_check=$(ls ${run_dir}/OutputDir/ | wc -w)
    if [[ $cleanup_check > 0 ]]; then
        # Check for completion of preprocessing
        preprocess_check_1=$(ls ${run_dir}/OutputDir/*_orig | wc -w)
        preprocess_check_2=$(ls ${run_dir}/OutputDir/GEOSChem.SpeciesConc*.nc4 | wc -w)
        if [[ $preprocess_check_1 != 14 || $preprocess_check_2 != 366 ]]; then
            echo "Preprocessing failed -- ${run_name}"

            if "$RestartProcesses"; then
                # Enter the run directory
                cd ${run_name}

                # Modify the run file
                sed -i -e "s@export OMP_NUM_THREADS@# export OMP_NUM_THREADS@g" \
                       -e "s@. ~/init/init@# . ~/init/init@g" \
                       -e "s@./geos@# ./geos@g" \
                       -e "s@echo@# echo@g" ${run_name}.run

                # Remove old output file and resubmit
                \rm preprocess_GC*.out
                ./${run_name}.run

                # Exit directory
                cd ..
            fi
        else
            # Check for completion of operator
            operator_check=$(ls ${run_dir}/ProcessedDir/ | wc -w)
            if [[ $operator_check != 365 ]]; then
                echo "Operator failed -- ${run_name}"

                if "$RestartProcesses"; then
                    # Enter the run directory
                    cd ${run_dir}

                    # Modify the run file
                    sed -i -e "s@export OMP_NUM_THREADS@# export OMP_NUM_THREADS@g" \
                           -e "s@. ~/init/init@# . ~/init/init@g" \
                           -e "s@./geos@# ./geos@g" \
                           -e "s@jid1=@# jid1=@g" \
                           -e "s@--dependency=afterok:\${jid1##\* } @@g" ${run_name}.run

                    # Remove old output files and resubmit
                    \rm -f TROPOMI_operator_*
                    ./${run_name}.run

                    # Exit directory
                    cd ..
                fi
            else
                echo "Clean up failed -- ${run_name}"
                if "$RestartProcesses"; then
                    \rm ${run_dir}/OutputDir/*
                fi
            fi
        fi
    else
        echo "Success -- ${run_name}"
    fi
fi

done

# Need to check that all files are removed afterward

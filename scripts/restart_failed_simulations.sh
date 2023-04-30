# Input options
RestartProcesses=${1}
Min=${2}
Max=${3}

# Directories
jac_dir="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs"

# Set up
cd $jac_dir

for x in $(seq ${Min} ${Max});
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
    check_pp_clean=$(ls ${run_dir}/OutputDir/ | wc -w)
    if [[ $check_pp_clean > 0 ]]; then
        # Check for completion of preprocessing
        ## Check that stratospheric data in 14 files are replaced
        [[ $(grep --no-filename 'Replacing data on' ${run_dir}/TROPOMI_operator_*.out | sort -u | wc -l) == 14 ]] && check_pp_strat=true || check_pp_strat=false
        [[ $(ls ${run_dir}/OutputDir/GEOSChem.SpeciesConc*.nc4 | wc -w) == 366 ]] && check_gc_count=true || check_gc_count=false
        if [[ ! $check_pp_strat || ! $check_gc_count ]]; then
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
            [[ $(ls ${run_dir}/ProcessedDir/ | wc -w) == 365 ]] && check_pp_count=true || check_pp_count=false
            # Check that the post-proocessing output is not size 0
            min_file_size=($(ls -lSh ${run_dir}/ProcessedDir | tail -n 1))
            min_file_size=${min_file_size[4]}
            [[ $min_file_size  == 208 ]] && check_pp_size=true || check_pp_size=false

            if [[ ! $check_pp_count || ! $check_pp_size ]]; then
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

for x in $(seq 14 105);
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

gcdir="TROPOMI_inversion_${xstr}"

# Check for GC completion
GC_check=$(tail -n 2 ${gcdir}/${gcdir}.log | head -n 1)
if [[ $GC_check != "**************   E N D   O F   G E O S -- C H E M   **************" ]]
then
    echo "GEOS-Chem failed -- ${gcdir}"
    # All GC runs worked in the last set of perturbations, so we're
    # skipping this part for now
fi

# Check for pre-processing completion
preprocess_check=$(tail -n 2 ${gcdir}/preprocess_GC*.out | head -n 1)

if [[ $preprocess_check != "Replaced data in 14 files." ]]
then
  echo "Preprocessing failed -- ${gcdir}"

  # Turn off switches in the run template
  sed -i -e "s:. ~/init/init"
fi

done

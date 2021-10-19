#!/bin/bash

# This script will set up CH4 analytical inversions with GEOS-Chem. See
# setup_ch4_inversion_instructions.txt for details (mps, 2/20/2020)

##=======================================================================
## User settings **MODIFY AS NEEDED**
##=======================================================================

# Turn on/off different steps. This will allow you to come back to this
# script and set up different stages later.
SetupTemplateRundir=true
SetupSpinupRun=false
SetupJacobianRuns=true
SetupInversion=false
SetupPosteriorRun=false
CompileCodeDir=false
CompileWithDebug=false
UseEvecPerts=true

# Set number of threads
export OMP_NUM_THREADS=16

# Path to inversion setup, post-processing code, and scripts
INV_PATH="$(dirname `pwd`)"
CODE_PATH="${INV_PATH}/python"
SCRIPT_PATH="${INV_PATH}/scripts"

# Name for this run
RUN_NAME="TROPOMI_inversion"

# Path where you want to set up CH4 inversion code and run directories
JAC_PATH="/n/holyscratch01/jacob_lab/hnesser"

# Path to find non-emissions input data
DATA_PATH="/n/holyscratch01/external_repos/GEOS-CHEM/gcgrid/gcdata/ExtData"

# Path to TROPOMI observations
TROP_PATH="/n/seasasfs02/CH4_inversion/InputData/Obs/TROPOMI"

# Path to data with accurate stratospheric concentrations
# and average vertical profiles
HALFSTEP_PATH="/n/seasasfs02/hnesser/TROPOMI_inversion/gc_outputs/"

# Path to code
GC_PATH="${HOME}/CH4_GC/Code.CH4_Inv"
GC_INPUTS_PATH="${INV_PATH}/GC_inputs"
#CODE_BRANCH="eigenvector_perturbations"

# Start and end date for the production simulations
START_DATE=20190101
END_DATE=20200101

# Start and end date for the spinup simulation
DO_SPINUP=false
SPINUP_START=20180401
SPINUP_END=20180501

# Path to initial restart file
RESTART_FILE="/n/seasasfs02/hnesser/TROPOMI_inversion/restarts/GEOSChem.Restart.${START_DATE}_0000z.nc4"

# Path to boundary condition files (for nested grid simulations)
# Must put backslash before $ in $YYYY$MM$DD to properly work in sed command
BC_FILES="/n/seasasfs02/hnesser/TROPOMI_inversion/boundary_conditions/GEOSChem.BoundaryConditions.\$YYYY\$MM\$DD_0000z.nc4"

# Jacobian settings
nPerturbationsMin=11
nPerturbationsMax=100
pPERT="1.0E-8"

# Path and file format for eigenvectors 
# (use evecnumevecnum to substitute for the number)
EVEC_PATH="/n/seasasfs02/hnesser/TROPOMI_inversion/inversion_data/eigenvectors"
EVEC_FILE="evec_pert_evecnumevecnum.nc"

# Grid settings (Nested NA)
RES="0.25x0.3125"
MET="geosfp"
LONS="-130.0 -60.0"
LATS=" 9.75 60.0"
HPOLAR="F"
LEVS="47"
NEST="T"
REGION="NA"  # NA,AS,CH,EU
BUFFER="3 3 3 3"

# Turn on observation operators and planeflight diagnostics?
GOSAT=false
TCCON=false
UseEmisSF=false
UseSeparateWetlandSF=false
UseOHSF=false
PLANEFLIGHT=false
PLANEFLIGHT_FILES="\/n\/seasasfs02\/hnesser\/GC_TROPOMI_bias\/PFs\/planeflight_combined_YYYYMMDD"
HourlyCH4=true

##=======================================================================
## Define met and grid fields for HEMCO_Config.rc
##=======================================================================
if [ "$MET" == "geosfp" ]; then
  metDir="GEOS_FP"
  native="0.25x0.3125"
elif [ "$MET" == "merra2" ]; then
  metDir="MERRA2"
  native="0.5x0.625"
fi
if [ "$RES" = "4x5" ]; then
    gridRes="4.0x5.0"
elif [ "$RES" == "2x2.5" ]; then
    gridRes="2.0x2.5"
else
    gridRes="$RES"
fi
if [ -z "$REGION" ]; then
    gridDir="$RES"
else
    gridDir="${RES}_${REGION}"
fi

##=======================================================================
## Get source code
##=======================================================================
# HON 2020/01/12: Removed submodule and added symbolic link
#cd ${GC_PATH}
#git checkout ${CODE_BRANCH}
cd ${JAC_PATH}
ln -sTf ${GC_PATH} GEOS-Chem

##=======================================================================
## Set up template run directory
##=======================================================================
if "$SetupTemplateRundir"; then

mkdir -p ${JAC_PATH}/${RUN_NAME}
cd ${JAC_PATH}/${RUN_NAME}
mkdir -p jacobian_runs

# Copy and update settings in inversion run scripts
cp ${SCRIPT_PATH}/run_jacobian_simulations.sh jacobian_runs/
sed -i -e "s:{RunName}:${RUN_NAME}:g" jacobian_runs/run_jacobian_simulations.sh
cp ${SCRIPT_PATH}/submit_jacobian_simulations_array.sh jacobian_runs/
sed -i -e "s:{START}:${nPerturbationsMin}:g" -e "s:{END}:${nPerturbationsMax}:g" jacobian_runs/submit_jacobian_simulations_array.sh

# Obtain GEOS-Chem input files: input.geos, HISTORY.rc, ch4_run, getRunInfo,
# Makefile, HEMCO_Diagn.rc, and HEMCO_Config.rc
RUN_TEMPLATE="template_run"
mkdir -p ${RUN_TEMPLATE}
cp -RLv ${GC_INPUTS_PATH}/input.geos.CH4 ${RUN_TEMPLATE}/input.geos
cp -RLv ${GC_INPUTS_PATH}/HISTORY.rc.CH4 ${RUN_TEMPLATE}/HISTORY.rc
cp -RLv ${SCRIPT_PATH}/GEOS-Chem_run.template ${RUN_TEMPLATE}
cp -RLv ${GC_INPUTS_PATH}/getRunInfo ${RUN_TEMPLATE}/
cp -RLv ${GC_INPUTS_PATH}/Makefile ${RUN_TEMPLATE}/
cp -RLv ${GC_INPUTS_PATH}/HEMCO_Diagn.rc.CH4 ${RUN_TEMPLATE}/HEMCO_Diagn.rc
cp -RLv ${GC_INPUTS_PATH}/HEMCO_Config.rc.CH4* ${RUN_TEMPLATE}/HEMCO_Config.rc

# Create run directory structure
cd $RUN_TEMPLATE
mkdir -p OutputDir
mkdir -p ProcessedDir
mkdir -p Restarts

### Update settings in GEOS-Chem_run.template
sed -i -e "s:invpathinvpath:${INV_PATH}:g" \
       -e "s:halfstephalfstep:${HALFSTEP_PATH}:g" \
       -e "s:satdirsatdir:${TROP_PATH}:g" GEOS-Chem_run.template

### Update settings in input.geos
sed -i -e "s:{DATE1}:${START_DATE}:g" \
       -e "s:{DATE2}:${END_DATE}:g" \
       -e "s:{TIME1}:000000:g" \
       -e "s:{TIME2}:000000:g" \
       -e "s:{MET}:${MET}:g" \
       -e "s:{DATA_ROOT}:${DATA_PATH}:g" \
       -e "s:{SIM}:CH4:g" \
       -e "s:{RES}:${gridRes}:g" \
       -e "s:{LON_RANGE}:${LONS}:g" \
       -e "s:{LAT_RANGE}:${LATS}:g" \
       -e "s:{HALF_POLAR}:${HPOLAR}:g" \
       -e "s:{NLEV}:${LEVS}:g" \
       -e "s:{NESTED_SIM}:${NEST}:g" \
       -e "s:{BUFFER_ZONE}:${BUFFER}:g" input.geos
if [ "$NEST" == "T" ]; then
    echo "Replacing timestep"
    sed -i -e "s|timestep \[sec\]: 600|timestep \[sec\]: 300|g" \
           -e "s|timestep \[sec\]: 1200|timestep \[sec\]: 600|g" input.geos
fi

# For CH4 inversions always turn analytical inversion on
OLD="Do analytical inversion?: F"
NEW="Do analytical inversion?: T"
sed -i "s/$OLD/$NEW/g" input.geos

# Turn other options on/off according to settings above
if "$GOSAT"; then
    OLD="Use GOSAT obs operator? : F"
    NEW="Use GOSAT obs operator? : T"
    sed -i "s/$OLD/$NEW/g" input.geos
fi
if "$TCCON"; then
    OLD="Use TCCON obs operator? : F"
    NEW="Use TCCON obs operator? : T"
    sed -i "s/$OLD/$NEW/g" input.geos
fi
if "$UseEmisSF"; then
    OLD=" => Use emis scale factr: F"
    NEW=" => Use emis scale factr: T"
    sed -i "s/$OLD/$NEW/g" input.geos
fi
if "$UseSeparateWetlandSF"; then
    OLD=" => Use sep. wetland SFs: F"
    NEW=" => Use sep. wetland SFs: T"
    sed -i "s/$OLD/$NEW/g" input.geos
fi
if "$UseOHSF"; then
    OLD=" => Use OH scale factors: F"
    NEW=" => Use OH scale factors: T"
    sed -i "s/$OLD/$NEW/g" input.geos
fi
if "$PLANEFLIGHT"; then
    mkdir -p Plane_Logs
    OLD="Turn on plane flt diag? : F"
    NEW="Turn on plane flt diag? : T"
    sed -i "s/$OLD/$NEW/g" input.geos
    OLD="Flight track info file  : Planeflight.dat.YYYYMMDD"
    NEW="Flight track info file  : ${PLANEFLIGHT_FILES}"
    sed -i "s/$OLD/$NEW/g" input.geos
    OLD="Output file name        : plane.log.YYYYMMDD"
    NEW="Output file name        : Plane_Logs\/plane.log.YYYYMMDD"
    sed -i "s/$OLD/$NEW/g" input.geos
fi

### Set up HEMCO_Config.rc
### Use monthly emissions diagnostic output for now
sed -i -e "s:End:Monthly:g" \
       -e "s:{VERBOSE}:0:g" \
       -e "s:{WARNINGS}:1:g" \
       -e "s:{DATA_ROOT}:${DATA_PATH}:g" \
       -e "s:{GRID_DIR}:${gridDir}:g" \
       -e "s:{MET_DIR}:${metDir}:g" \
       -e "s:{NATIVE_RES}:${native}:g" \
       -e "s:\$ROOT/SAMPLE_BCs/v2019-05/CH4/GEOSChem.BoundaryConditions.\$YYYY\$MM\$DD_\$HH\$MNz.nc4:${BC_FILES}:g" HEMCO_Config.rc
if [ ! -z "$REGION" ]; then
    sed -i -e "s:\$RES:\$RES.${REGION}:g" HEMCO_Config.rc
fi

### Set up HISTORY.rc

# Change restart frequency to monthly
sed -i -e "s/Restart.frequency:          'End'/Restart.frequency:          00000100 000000/g" \
       -e "s/Restart.duration:           'End'/Restart.duration:           00000100 000000/g" HISTORY.rc

#Use monthly output for now
sed -i -e "s:{FREQUENCY}:00000100 000000:g" \
       -e "s:{DURATION}:00000100 000000:g" \
       -e 's:'\''CH4:#'\''CH4:g' HISTORY.rc

# If turned on, save out hourly CH4 concentrations and pressure fields to
# daily files
if "$HourlyCH4"; then
    # Turn on proper collections
    sed -i -e 's/SpeciesConc.frequency:      00000100 000000/SpeciesConc.frequency:      00000000 010000/g' \
	   -e 's/SpeciesConc.duration:       00000100 000000/SpeciesConc.duration:       00000001 000000/g' \
           -e 's/SpeciesConc.mode:           '\''time-averaged/SpeciesConc.mode:           '\''instantaneous/g' HISTORY.rc

    # Turn off superfluous fields
    sed -i -e 's/'\''Met\_/\# '\''Met\_/g' HISTORY.rc

    # Turn back on needed fields
    sed -i -e 's/\# '\''Met\_DELPDRY/'\''Met\_DELPDRY/g' \
           -e 's/\# '\''Met\_PS1WET/'\''Met\_PS1WET/g' \
           -e 's/\# '\''Met\_PS1DRY/'\''Met\_PS1DRY/g' \
           -e 's/\# '\''Met\_SPHU1/'\''Met\_SPHU1/g' \
           -e 's/\# '\''Met\_TMPU1/'\''Met\_TMPU1/g' \
           -e 's/\# '\''Met\_PEDGE/'\''Met\_PEDGE/g' \
           -e 's/\# '\''Met\_PEDGEDRY/'\''Met\_PEDGEDRY/g' HISTORY.rc
fi

if "$CompileCodeDir"; then
    cd ${JAC_PATH}/${RUN_NAME}/${RUN_TEMPLATE} 

    ### Compile GEOS-Chem and store executable in template run directory
    make realclean CODE_DIR=${JAC_PATH}/GEOS-Chem
    if ! "${CompileWithDebug}"; then
	if "$PLANEFLIGHT"; then
	    make -j${OMP_NUM_THREADS} build CODE_DIR=${JAC_PATH}/GEOS-Chem BPCH_DIAG=y
	else
	    make -j${OMP_NUM_THREADS} build CODE_DIR=${JAC_PATH}/GEOS-Chem
	fi
    else
	make -j${OMP_NUM_THREADS} build CODE_DIR=${JAC_PATH}/GEOS-Chem BPCH_DIAG=y DEBUG=y BOUNDS=y TRACEBACK=y FPE=y
    fi

    cp geos ${GC_INPUTS_PATH}/
    cp compile.log ${GC_INPUTS_PATH}/
    cp lastbuild ${GC_INPUTS_PATH}/
else
    cp ${GC_INPUTS_PATH}/geos .
    cp ${GC_INPUTS_PATH}/compile.log .
    cp ${GC_INPUTS_PATH}/lastbuild .
fi

### Navigate back to top-level directory
cd ..

fi # SetupTemplateRunDir

##=======================================================================
##  Set up spinup run directory
##=======================================================================
if  "$SetupSpinupRun"; then

    ### Define the run directory name
    spinup_name="${RUN_NAME}_Spinup"

    ### Make the directory
    runDir="spinup_run"
    mkdir -p ${runDir}

    ### Copy and point to the necessary data
    cp -r ${RUN_TEMPLATE}/*  ${runDir}
    cd $runDir

    ### Link to GEOS-Chem executable instead of having a copy in each run dir
    rm -rf geos
    ln -sTf ../${RUN_TEMPLATE}/geos .

    # Link to restart file
    ln -sTf $RESTART_FILE GEOSChem.Restart.${SPINUP_START}_0000z.nc4
    
    ### Update settings in input.geos
    sed -i -e "s|${START_DATE}|${SPINUP_START}|g" \
           -e "s|${END_DATE}|${SPINUP_END}|g" \
	   -e "s|Do analytical inversion?: T|Do analytical inversion?: F|g" \
	   -e "s|pertpert|1.0|g" \
           -e "s|clustnumclustnum|0|g" input.geos

    ### Create run script from template
    sed -e "s:namename:${spinup_name}:g" \
	-e "s:##:#:g" GEOS-Chem_run.template > ${spinup_name}.run
    chmod 755 ${spinup_name}.run

    ### Print diagnostics
    echo "CREATED: ${runDir}"
    
    ### Navigate back to top-level directory
    cd ..
    
fi # SetupSpinupRun

##=======================================================================
##  Set up posterior run directory
##=======================================================================
if  "$SetupPosteriorRun"; then

    ### Define the run directory name
    posterior_name="${RUN_NAME}_Posterior"

    ### Make the directory
    runDir="posterior_run"
    mkdir -p ${runDir}

    ### Copy and point to the necessary data
    cp -r ${RUN_TEMPLATE}/*  ${runDir}
    cd $runDir

    ### Link to GEOS-Chem executable instead of having a copy in each run dir
    rm -rf geos
    ln -sTf ../${RUN_TEMPLATE}/geos .

    # Link to restart file
    if "$DO_SPINUP"; then
       ln -s ../../spinup_run/GEOSChem.Restart.${SPINUP_END}_0000z.nc4 GEOSChem.Restart.${START_DATE}_0000z.nc4
    else
       ln -sTf $RESTART_FILE GEOSChem.Restart.${START_DATE}_0000z.nc4
    fi
    
    ### Update settings in input.geos
    sed -i -e "s|Do analytical inversion?: T|Do analytical inversion?: F|g" \
	   -e "s|pertpert|1.0|g" \
           -e "s|clustnumclustnum|0|g" input.geos

    ### Create run script from template
    sed -e "s:namename:${spinup_name}:g" \
	-e "s:##:#:g" GEOS-Chem_run.template > ${posterior_name}.run
    chmod 755 ${posterior_name}.run

    ### Print diagnostics
    echo "CREATED: ${runDir}"
    echo "\nNote: You will need to manually modify HEMCO_Config.rc to apply the appropriate scale factors."
    
    ### Navigate back to top-level directory
    cd ..
    
fi # SetupPosteriorRun

##=======================================================================
##  Set up Jacobian run directories
##=======================================================================
if "$SetupJacobianRuns"; then

cd ${JAC_PATH}/${RUN_NAME}
pwd
# Initialize (x=0 is base run, i.e. no perturbation; x=1 is cluster=1; etc.)
x=$nPerturbationsMin

# Create run directory for each cluster so we can apply perturbation to each
while [[ $x -le $nPerturbationsMax && $x -ge $nPerturbationsMin ]];do

   ### Positive or negative perturbation
   PERT=$pPERT
   xUSE=$x

   ### Add zeros to string name
   if [ $x -lt 10 ]; then
      xstr="000${x}"
   elif [ $x -lt 100 ]; then
      xstr="00${x}"
   elif [ $x -lt 1000 ]; then
      xstr="0${x}"
   else
      xstr="${x}"
   fi

   ### Define the run directory name
   name="${RUN_NAME}_${xstr}"

   ### Make the directory
   runDir="./jacobian_runs/${name}"
   mkdir -p ${runDir}

   ### Copy and point to the necessary data
   echo cp -r ${RUN_TEMPLATE}/*  ${runDir}
   # cd $runDir

   # ### Link to GEOS-Chem executable instead of having a copy in each run dir
   # rm -rf geos
   # ln -sTf ../../${RUN_TEMPLATE}/geos geos

   # # Link to restart file
   # if "$DO_SPINUP"; then
   #     ln -s ../../spinup_run/GEOSChem.Restart.${SPINUP_END}_0000z.nc4 GEOSChem.Restart.${START_DATE}_0000z.nc4
   # else
   #     ln -sTf $RESTART_FILE GEOSChem.Restart.${START_DATE}_0000z.nc4
   # fi
   
   # ### Update settings in input.geos
   # sed -i -e "s:pertpert:${PERT}:g" \
   #        -e "s:clustnumclustnum:${xUSE}:g" input.geos

   # # Turn on LevelEdgeDiags for the prior run
   # if [ $x -eq 0 ]; then
   #     sed -i -e 's/#'\''LevelEdgeDiags/'\''LevelEdgeDiags/g' \
	  #     -e 's/LevelEdgeDiags.frequency:   00000100 000000/LevelEdgeDiags.frequency:   00000000 010000/g' \
	  #     -e 's/LevelEdgeDiags.duration:    00000100 000000/LevelEdgeDiags.duration:    00000001 000000/g' \
	  #     -e 's/LevelEdgeDiags.mode:        '\''time-averaged/LevelEdgeDiags.mode:        '\''instantaneous/g' HISTORY.rc

   #     JAC="False"
   # fi

   # # Update settings for non-prior runs
   # if [ $x -gt 0 -a "${UseEvecPerts}" ]; then
   #     ### Update settings in input.geos
   #     # Turn on eigenvector perturbations
   #     OLD="Eigenvector pert.?  : F"
   #     NEW="Eigenvector pert.?  : T"
   #     sed -i "s/$OLD/$NEW/g" input.geos

   #     ### Update settings in HEMCO_Config.rc
   #     # Section extension switches
   #     OLD="EVECS                  :       false"
   #     NEW="EVECS                  :       true"
   #     sed -i "s/$OLD/$NEW/g" HEMCO_Config.rc

   #     # Eigenvector file
   #     OLD="/n/seasasfs02/hnesser/TROPOMI_inversion/evec_perturbations/evec_perturbations_evecnumevecnum.nc"
   #     NEW="${EVEC_PATH}/${EVEC_FILE//evecnumevecnum/${xstr}}"
   #     sed -i "s:$OLD:$NEW:g" HEMCO_Config.rc

   #     ### Update HEMCO_Diagn.rcx
   #     NEW="EmisCH4_Evecs        CH4    0   15   -1   2   kg/m2/s  CH4_emissions_from_eigenvectors"
   #     sed -i "/\#EOC/i$NEW" HEMCO_Diagn.rc

   #     JAC="TRUE"
   # fi

   # ### Create run script from template
   # sed -e "s:namename:${name}:g" \
   #     -e "s:jacjac:${JAC}:g" GEOS-Chem_run.template > ${name}.run
   # chmod 755 ${name}.run
   # rm GEOS-Chem_run.template

   # ### Navigate back to top-level directory
   # cd ../..

   # ### Increment
   # x=$[$x+1]

   # ### Print diagnostics
   # echo "CREATED: ${runDir}"

done

echo "=== DONE CREATING JACOBIAN RUN DIRECTORIES ==="

fi  # SetupJacobianRuns

##=======================================================================
##  Setup inversion directory -- THIS IS NOT RELEVANT TO MY USE CASE
##=======================================================================
if "$SetupInversion"; then

    cd ${JAC_PATH}/$RUN_NAME
    mkdir -p inversion
    mkdir -p inversion/data_converted
    mkdir -p inversion/data_GC
    mkdir -p inversion/Sensi
    ln -s /n/holylfs/LABS/jacob_lab/lshen/CH4/TROPOMI/data inversion/data_TROPOMI
    cp ${INV_PATH}/PostprocessingScripts/CH4_TROPOMI_INV/*.py inversion/
    cp ${INV_PATH}/PostprocessingScripts/CH4_TROPOMI_INV/run_inversion.sh inversion/
    sed -i -e "s:{CLUSTERS}:${nClusters}:g" \
	   -e "s:{START}:${START_DATE}:g" \
           -e "s:{END}:${END_DATE}:g" \
	   -e "s:{RUNDIRS}:${JAC_PATH}/${RUN_NAME}/jacobian_runs:g" \
	   -e "s:{RUNNAME}:${RUN_NAME}:g" \
	   -e "s:{MYPATH}:${JAC_PATH}:g" inversion/run_inversion.sh
	   
    echo "=== DONE SETTING UP INVERSION DIRECTORY ==="

fi #SetupInversion

exit 0

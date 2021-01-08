#!/bin/bash

# ------------------------------------------------------------------
### Set up GEOS-Chem for Jacobian run (mps, 2/20/2020)
# ------------------------------------------------------------------

##=======================================================================
## Set variables

# HON 2020/07/15 - Add variables for UT and code directory
CODE_PATH="${HOME}/CH4_GC"
GC_NAME="Code.CH4_Inv"
UT_NAME="UnitTester.CH4_Inv"

#MREW 2020/11/11
GC_VERSION=12.7.1

# Path where you want to set up CH4 inversion code and run directories
SCRATCH_PATH="/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion"

# Path to find non-emissions input data
DATA_PATH="/n/holylfs/EXTERNAL_REPOS/GEOS-CHEM/gcgrid/data/ExtData"

# Path to run scripts
RUN_SCRIPTS="${HOME}/TROPOMI_inversion/geoschem"
#RUN_SCRIPTS="/n/seasasfs02/CH4_inversion/RunDirScripts"

# Start and end date fo the simulations
START_DATE=20190101
END_DATE=20180601

# Path to initial restart file
RESTART_FILE="/n/seasasfs02/hnesser/GC_TROPOMI_bias/restarts/GEOSChem.Restart.${START_DATE}_0000z.nc4"

# Path to boundary condition files (for nested grid simulations)
# Must put backslash before $ in $YYYY$MM$DD to properly work in sed command
BC_FILES="/n/seasasfs02/hnesser/GC_TROPOMI_bias/BCs/GEOSChem.BoundaryConditions.\$YYYY\$MM\$DD_0000z.nc4"

# Grid settings
RES="0.25x0.3125"
MET="geosfp"
LONS="-130.0 -60.0"
LATS=" 9.75 60.0"
HPOLAR="F"
LEVS="47"
NEST="T"
REGION="NA"  # Nested grid region (NA,AS,CH,EU); Leave blank for global or custom
BUFFER="3 3 3 3"

# Jacobian settings
START_I=0
END_I=0
pPERT="1.0" #1.5
RUN_NAME="CH4_Jacobian"
RUN_TEMPLATE="${RES}_template"

# Copy clean UT and Code directory? Only needed on first setup
CopyDirs=true

# Turn on observation operators and planeflight diagnostics?
GOSAT=true
TCCON=false
UseEmisSF=false
UseSeparateWetlandSF=false
UseOHSF=false
PLANEFLIGHT=true
PLANEFLIGHT_FILES="\/n\/seasasfs02\/hnesser\/GC_TROPOMI_bias\/PFs\/planeflight_combined_YYYYMMDD"
HourlyCH4=true

### Number of Clusters
start=$START_I
stop=$END_I
x=$start

##=======================================================================
## Get source code and run directories
#if "$CopyDirs"; then
#    # Copy source code with CH4 analytical inversion updates to your space
#    # Make sure branch with latest CH4 inversion updates is checked out
#    cp -r /n/seasasfs02/CH4_inversion/Code.CH4_Inv .
#    cd Code.CH4_Inv
#    git checkout CH4_Analytical_Inversion
#    cd ..
#
#    # Copy Unit Tester to create run directory to your space
#    # Make sure branch with latest CH4 inversion updates is checked out
#    cp -r /n/seasasfs02/CH4_inversion/UnitTester.CH4_Inv .
#    cd UnitTester.CH4_Inv
#    git checkout CH4_Analytical_Inversion
#    cd ..
#fi

# HON 2020/01/05: Note: this should be written so that we could run it from within my
# TROPOMI_inversion directory

# Copy

# Conduct all this work from within SCRATCH_PATH
cd ${SCRATCH_PATH}

# Copy source code with CH4 analytical inversion updates to your space
# Make sure branch with latest CH4 inversion updates is checked out
# HON 2020/07/15 - Add an if statement that copies the code if the
# code directory is not specified and creates a symbolic link to the
# existing code directory otherwise.
if [[ -d "${CODE_PATH}/${GC_NAME}" ]]
then
    echo "Code directory already exists."
    echo "Did you check that your directory is up to date?"
    ln -s -f ${CODE_PATH}/${GC_NAME} ./Code.CH4_Inv
else
    echo "Cloning seasasfs02 code directory."
    cp -r /n/seasasfs02/CH4_inversion/Code.CH4_Inv ${CODE_PATH}
    cd ${CODE_PATH}/Code.CH4_Inv
    git checkout CH4_Analytical_Inversion
    cd ${SCRATCH_PATH}
    ln -s ${CODE_PATH}/Code.CH4_Inv ./Code.CH4_Inv
fi

# Copy Unit Tester to create run directory to your space
# Make sure branch with latest CH4 inversion updates is checked out
# HON 2020/07/15 - Add an if statement that copies the UT if the
# UT directory is not specified and creates a symbolic link to the
# existing UT directory otherwise.
if [[ -d "${CODE_PATH}/${UT_NAME}" ]]
then
    echo "Unit tester already exists."
    ln -s -f ${CODE_PATH}/${UT_NAME} ./UnitTester.CH4_Inv
else
    echo "Cloning seasasfs02 unit tester."
    cp -r /n/seasasfs02/CH4_inversion/UnitTester.CH4_Inv ${CODE_PATH}
    cd ${CODE_PATH}/UnitTester.CH4_Inv
    git checkout CH4_Analytical_Inversion
    cd ${SCRATCH_PATH}
    ln -s -f ${CODE_PATH}/UnitTester.CH4_Inv
fi

##=======================================================================
## Copy run directory with template files directly from unit tester

# Create run directory folder and copy bash scripts that run
# the Jacobian construction
echo "----------------------"
pwd
echo "----------------------"
# mkdir -p $RUN_NAME
# cd $RUN_NAME
# mkdir -p run_dirs
# cp ${RUN_SCRIPTS}/submit_array_jobs run_dirs/
# sed -i -e "s:{RunName}:${RUN_NAME}:g" run_dirs/submit_array_jobs
# cp ${RUN_SCRIPTS}/run_array_job run_dirs/
# sed -i -e "s:{START}:${START_I}:g" -e "s:{END}:${END_I}:g" run_dirs/run_array_job
# cp ${RUN_SCRIPTS}/rundir_check.sh run_dirs/
# mkdir -p bin
# if [ "$NEST" == "T" ]; then
#   cp -rLv ${SCRATCH_PATH}/UnitTester.CH4_Inv/runs/${MET}_*_CH4_na $RUN_TEMPLATE
# else
#   cp -rLv ${SCRATCH_PATH}/UnitTester.CH4_Inv/runs/${RES}_CH4 $RUN_TEMPLATE
# fi

# # Set up template run directory
# echo "========================================="
# echo "=== Setting up template run directory ==="
# cd $RUN_TEMPLATE
# cp ${SCRATCH_PATH}/UnitTester.CH4_Inv/runs/shared_inputs/Makefiles/Makefile .
# cp ${SCRATCH_PATH}/UnitTester.CH4_Inv/perl/getRunInfo .
# cp ${RUN_SCRIPTS}/run.template .
# ln -s -f $RESTART_FILE .
# mkdir -p OutputDir
# cd ..
# echo "========================================="

# # Define met and grid fields for HEMCO_Config.rc
# if [ "$MET" == "geosfp" ]; then
#   metDir="GEOS_FP"
#   native="0.25x0.3125"
# elif [ "$MET" == "merra2" ]; then
#   metDir="MERRA2"
#   native="0.5x0.625"
# fi
# if [ "$RES" = "4x5" ]; then
#     gridRes="4.0x5.0"
# elif [ "$RES" == "2x2.5" ]; then
#     gridRes="2.0x2.5"
# else
#     gridRes="$RES"
# fi
# if [ -z "$REGION" ]; then
#     gridDir="$RES"
# else
#     gridDir="${RES}_${REGION}"
# fi

# ##=======================================================================
# ##  Create run directories
# echo "================================"
# echo "=== Creating run directories ==="
# while [ $x -le $stop ];do

#    # All of this will be moot when I use eigenvector perturbations
#    ### Positive or negative perturbation
#    if [ $x -eq -1 ]; then
#       PERT="1.0"
#       xUSE=$x
#    else
#       PERT=$pPERT
#       xUSE=$x
#    fi

#    ### Add zeros to string name
#    if [ $x -lt 10 ]; then
#       xstr="000${x}"
#    elif [ $x -lt 100 ]; then
#       xstr="00${x}"
#    elif [ $x -lt 1000 ]; then
#       xstr="0${x}"
#    else
#       xstr="${x}"
#    fi

#    ### Define the run directory name
#    name="${RUN_NAME}_${xstr}"

#    ### Make the directory
#    echo "===== Creating run directory ${x}"
#    runDir="./run_dirs/${name}"
#    mkdir -p ${runDir}
#    mkdir -p ${runDir}/Plane_Logs
#    mkdir -p ${runDir}/Restarts

#    ### Copy and point to the necessary data
#    cp -r ${RUN_TEMPLATE}/*  ${runDir}
#    cd $runDir

#    ### Create input.geos file from template
#    echo "======= Modifying input.geos"
#    InputFile="input.geos.template"
#    sed -e "s:{DATE1}:${START_DATE}:g" \
#        -e "s:{DATE2}:${END_DATE}:g" \
#        -e "s:{TIME1}:000000:g" \
#        -e "s:{TIME2}:000000:g" \
#        -e "s:{MET}:${MET}:g" \
#        -e "s:{DATA_ROOT}:${DATA_PATH}:g" \
#        -e "s:{SIM}:CH4:g" \
#        -e "s:{RES}:${gridRes}:g" \
#        -e "s:{LON_RANGE}:${LONS}:g" \
#        -e "s:{LAT_RANGE}:${LATS}:g" \
#        -e "s:{HALF_POLAR}:${HPOLAR}:g" \
#        -e "s:{NLEV}:${LEVS}:g" \
#        -e "s:{NESTED_SIM}:${NEST}:g" \
#        -e "s:{BUFFER_ZONE}:${BUFFER}:g" \
#        -e "s:pertpert:${PERT}:g" \
#        -e "s:clustnumclustnum:${xUSE}:g" \
#        $InputFile > input.geos.temp
#    mv input.geos.temp input.geos
#    rm input.geos.template

#    # For CH4 inversions always turn analytical inversion on
#    OLD="Do analytical inversion?: F"
#    NEW="Do analytical inversion?: T"
#    sed -i "s/$OLD/$NEW/g" input.geos

#    OLD="L (ND65) diag?: T"
#    NEW="L (ND65) diag?: F"
#    sed -i "s/$OLD/$NEW/g" input.geos

#    if "$GOSAT"; then
#        OLD="Use GOSAT obs operator? : F"
#        NEW="Use GOSAT obs operator? : T"
#        sed -i "s/$OLD/$NEW/g" input.geos
#    fi
#    if "$TCCON"; then
#        OLD="Use TCCON obs operator? : F"
#        NEW="Use TCCON obs operator? : T"
#        sed -i "s/$OLD/$NEW/g" input.geos
#    fi
#    if "$UseEmisSF"; then
#        OLD=" => Use emis scale factr: F"
#        NEW=" => Use emis scale factr: T"
#        sed -i "s/$OLD/$NEW/g" input.geos
#    fi
#    if "$UseSeparateWetlandSF"; then
#        OLD=" => Use sep. wetland SFs: F"
#        NEW=" => Use sep. wetland SFs: T"
#        sed -i "s/$OLD/$NEW/g" input.geos
#    fi
#    if "$UseOHSF"; then
#        OLD=" => Use OH scale factors: F"
#        NEW=" => Use OH scale factors: T"
#        sed -i "s/$OLD/$NEW/g" input.geos
#    fi
#    if "$PLANEFLIGHT"; then
#        OLD="Turn on plane flt diag? : F"
#        NEW="Turn on plane flt diag? : T"
#        sed -i "s/$OLD/$NEW/g" input.geos
#        OLD="Flight track info file  : Planeflight.dat.YYYYMMDD"
#        NEW="Flight track info file  : ${PLANEFLIGHT_FILES}"
#        sed -i "s/$OLD/$NEW/g" input.geos
#        OLD="Output file name        : plane.log.YYYYMMDD"
#        NEW="Output file name        : Plane_Logs\/plane.log.YYYYMMDD"
#        sed -i "s/$OLD/$NEW/g" input.geos
#    fi

#    # FAST-JX directory
#    OLD="v2019-10"
#    NEW="v2019-04"
#    sed -i "s/$OLD/$NEW/g" input.geos

#    ### Set up HEMCO_Config.rc
#    echo "======= Modifying HEMCO_Config.rc"
#    ### Use monthly emissions diagnostic output for now
#    sed -e "s:End:Monthly:g" \
#        -e "s:{VERBOSE}:0:g" \
#        -e "s:{WARNINGS}:1:g" \
#        -e "s:{DATA_ROOT}:${DATA_PATH}:g" \
#        -e "s:{GRID_DIR}:${gridDir}:g" \
#        -e "s:{MET_DIR}:${metDir}:g" \
#        -e "s:{NATIVE_RES}:${native}:g" \
#        -e "s:\$ROOT/SAMPLE_BCs/v2019-05/CH4/GEOSChem.BoundaryConditions.\$YYYY\$MM\$DD_\$HH\$MNz.nc4:${BC_FILES}:g" \
#        HEMCO_Config.template > HEMCO_Config.rc
#    rm HEMCO_Config.template
#    if [ ! -z "$REGION" ]; then
#        sed -i -e "s:\$RES:\$RES.${REGION}:g" HEMCO_Config.rc
#    fi

#    # adding in CanMexTia
#    sed -i '61 a \ \ \ \ --> CanMexTia              :       true' HEMCO_Config.rc
#    sed -i '/)))GFEI/ r ${RUN_SCRIPTS}/CanMexTia_text.txt' HEMCO_Config.rc
#    sed -i '904r ${RUN_SCRIPTS}/CanMexTiaMASK_text.txt' HEMCO_Config.rc

#    # removing EDGARv432 oil and gas
#    OLD="0 CH4_OILGAS"
#    NEW="#0 CH4_OILGAS"
#    sed -i "s/$OLD/$NEW/g" HEMCO_Config.rc

#    # using same JPL_WETCHARTS as xlu
#    OLD="v2020-04\/JPL_WetCharts\/JPL_WetCharts_\$YYYY.Ensemble_Mean.0.5x0.5.nc emi_ch4 2009-2017"
#    NEW="v2020-09\/JPL_WetCharts\/HEensemble\/products\/JPL_WetCharts_\$YYYY.Ensemble_Mean.0.5x0.5.nc emi_ch4 2010-2019"
#    sed -i "s/$OLD/$NEW/g" HEMCO_Config.rc

#    # using same QFED2 as xlu
#    OLD="v2020-04\/GFED"
#    NEW="v2020-09\/GFED"
#    sed -i "s/$OLD/$NEW/g" HEMCO_Config.rc

#    # getting restart data
#    OLD='SPC_           .\/GEOSChem.Restart.\$YYYY\$MM\$DD_\$HH\$MNz.nc4 SpeciesRst_?ALL?    \$YYYY\/\$MM\/\$DD\/\$HH EY  xyz 1 * - 1 1'
#    NEW='SPC_           .\/restarts_\$YYYY\/GEOSChem_restart.\$YYYY\$MM\$DD\$HH\$MN.nc SPC_?ALL?           \$YYYY\/\$MM\/\$DD\/\$HH EY  xyz 1 * - 1 1'
#    sed -i "s/$OLD/$NEW/g" HEMCO_Config.rc

#    ### Set up HISTORY.rc
#    echo "======= Modifying HISTORY.rc"
#    ### use monthly output for now
#    sed -e "s:{FREQUENCY}:00000000 010000:g" \
#        -e "s:{DURATION}:00000001 000000:g" \
#        -e 's:'\''CH4:#'\''CH4:g' \
#        HISTORY.rc.template > HISTORY.rc
#    rm HISTORY.rc.template

#    # If turned on, save out hourly CH4 concentrations and pressure fields to
#    # daily files
#    if "$HourlyCH4"; then
#        sed -e 's/SpeciesConc.frequency:      00000100 000000/SpeciesConc.frequency:      00000000 010000/g' \
# 	   -e 's/SpeciesConc.duration:       00000100 000000/SpeciesConc.duration:       00000001 000000/g' \
# 	   -e 's/SpeciesConc.mode:           '\''time-averaged/SpeciesConc.mode:           '\''instantaneous/g' \
# 	   -e 's/#'\''LevelEdgeDiags/'\''LevelEdgeDiags/g' \
# 	   -e 's/LevelEdgeDiags.frequency:   00000100 000000/LevelEdgeDiags.frequency:   00000000 010000/g' \
# 	   -e 's/LevelEdgeDiags.duration:    00000100 000000/LevelEdgeDiags.duration:    00000001 000000/g' \
# 	   -e 's/LevelEdgeDiags.mode:        '\''time-averaged/LevelEdgeDiags.mode:        '\''instantaneous/g' HISTORY.rc > HISTORY.temp
#        mv HISTORY.temp HISTORY.rc
#    fi

#    OLD="#'StateMet',"
#    NEW="'StateMet',"
#    sed -i "s/$OLD/$NEW/g" HISTORY.rc

#    OLD="'%y4%m2%d2_%h2%n2z.nc4'"
#    NEW="'%y4%m2%d2.%h2%n2z.nc4'"
#    sed -i "s/$OLD/$NEW/g" HISTORY.rc

#    OLD="time-averaged"
#    NEW="instantaneous"
#    sed -i "s/$OLD/$NEW/g" HISTORY.rc

#    ### Create run script from template
#    sed -e "s:namename:${name}:g" \
#        run.template > ${name}.run
#    rm run.template
#    chmod 755 ${name}.run

#    ### Compile code when creating first run directory
#    if [ $x -eq $START_I ]; then
#        echo "======= Compiling GEOS-Chem"
#        echo ${CODE_PATH}/${GC_NAME}
#        make realclean CODE_PATH=../${GC_NAME}
#        make -j${OMP_NUM_THREADS} build BPCH_DIAG=y CODE_PATH=../${GC_NAME}
#        cp -av geos ../../bin/
#    else
#        ln -s -f ../../bin/geos .
#    fi

#    ### Navigate back to top-level directory
#    cd ../..

#    ### Increment
#    x=$[$x+1]

#    ### Print diagnostics
#    echo "=== CREATED: ${runDir}"
#    echo "================================"

# done

# echo "=== DONE CREATING JACOBIAN RUN DIRECTORIES ==="

# exit 0

#!/bin/bash

module load Anaconda3/5.0.1-fasrc01
source activate ~/python/miniconda/envs/TROPOMI_inversion

export DASK_DISTRIBUTED__DASHBOARD__LINK="proxy/{port}/status"

/n/home04/hnesser/TROPOMI_inversion/scripts/launch_dask_scheduler.sh
/n/home04/hnesser/TROPOMI_inversion/scripts/launch_dask_worker.sh
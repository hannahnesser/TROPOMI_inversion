#!/bin/bash

# Edit apply_TROPOMI_operator.sh to change settings.

sbatch --array=3-4 apply_TROPOMI_operator.sh

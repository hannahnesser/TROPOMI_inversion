#!/bin/sh
# Script to preload ESMF dynamic trace library
env LD_PRELOAD="$LD_PRELOAD /Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/env/lib/libesmftrace_preload.dylib" $*

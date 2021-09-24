if __name__ == '__main__':
    import sys
    import xarray as xr
    # import dask.array as da
    import numpy as np
    import pandas as pd

    import glob

    ## ---------------------------------------------------------------------##
    ## Set user preferences
    ## ---------------------------------------------------------------------##
    # Cannon
    prior_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final'
    perturbation_dirs = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_????'
    code_dir = '/n/home04/hnesser/TROPOMI_inversion/python/'
    # data_dir = sys.argv[...]
    # code_dir = sys.argv[...]

    # Import custom packages
    import sys
    sys.path.append(code_dir)
    import gcpy as gc
    import inversion_settings as s

    ## ---------------------------------------------------------------------##
    ## Functions
    ## ---------------------------------------------------------------------##
    def get_model_ch4(file_list):
        data = np.array([])
        for f in file_list:
            data = np.concatenate((data, gc.load_obj(f)[:, 1]))
        return data

    ## ---------------------------------------------------------------------##
    ## Create list of perturbation directories
    ## ---------------------------------------------------------------------##
    perturbation_dirs = glob.glob(perturbation_dirs)
    perturbation_dirs = [p for p in perturbation_dirs
                         if p.split('_')[-1] != '0000']
    perturbation_dirs.sort()

    ## ---------------------------------------------------------------------##
    ## Iterate through months and files since we must make monthly Jacobians
    ## ---------------------------------------------------------------------##
    for m in s.months:
        ## -----------------------------------------------------------------##
        ## Load the data for the prior simulation
        ## -----------------------------------------------------------------##
        prior_files = glob.glob(f'{prior_dir}/ProcessedDir/{s.year:04d}{m:02d}??_GCtoTROPOMI.pkl')
        prior_files.sort()
        prior = get_model_ch4(prior_files)

        ## -----------------------------------------------------------------##
        ## Iterate through the perturbation directories
        ## -----------------------------------------------------------------##
        # Make a Jacobian

        for p in perturbation_dirs:
            # Load files
            pert_files = glob.glob(f'{p}/ProcessedDir/{s.year:04d}{m:02d}??_GCtoTROPOMI.pkl')
            pert_files.sort()
            pert = get_model_ch4(pert_files)

            # Get the Jacobian column
            diff = pert - prior





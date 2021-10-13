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
    # prior_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final'
    # perturbation_dirs = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_????'
    # data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion'
    # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
    month = int(sys.argv[1])
    prior_dir = sys.argv[2]
    perturbation_dirs = sys.argv[3:-2]
    data_dir = sys.argv[-2]
    code_dir = sys.argv[-1]

    # Import custom packages
    import sys
    sys.path.append(code_dir)
    import gcpy as gc
    import inversion_settings as s

    ## -------------------------------------------------------------------- ##
    ## Set up a dask client
    ## -------------------------------------------------------------------- ##
    from dask.distributed import Client, LocalCluster, progress
    from dask.diagnostics import ProgressBar
    import dask.config
    import dask.array as da
    dask.config.set({'distributed.comm.timeouts.connect' : 90,
                     'distributed.comm.timeouts.tcp' : 150,
                     'distributed.adaptive.wait-count' : 90,
                     'array.slicing.split_large_chunks' : False,
                     'temporary_directory' : f'{data_dir}/dask-worker-space-{month}'})
    nstate_chunk = 1e3 # int(np.sqrt(max_chunk_size)/5)
    nobs_chunk = 4e4 # int(max_chunk_size/nstate_chunk/5)

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
    # perturbation_dirs = glob.glob(perturbation_dirs)
    # perturbation_dirs = [p for p in perturbation_dirs
    #                      if p.split('_')[-1] != '0000']
    perturbation_dirs.sort()

    # Set state chunks
    nvec_chunk = len(perturbation_dirs)

    ## ---------------------------------------------------------------------##
    ## Load and subset the reduction operator
    ## ---------------------------------------------------------------------##
    reduction = np.load(f'{data_dir}/reduction0.npy')
    reduction = reduction[:len(perturbation_dirs), :]
    reduction = xr.DataArray(reduction, dims=['nvec', 'nstate'])
    reduction = reduction.chunk(chunks={'nvec' : nvec_chunk,
                                        'nstate' : nstate_chunk})

    ## ---------------------------------------------------------------------##
    ## Load the data for the prior simulation
    ## ---------------------------------------------------------------------##
    prior_files = glob.glob(f'{prior_dir}/ProcessedDir/{s.year:04d}{month:02d}??_GCtoTROPOMI.pkl')
    prior_files.sort()
    prior = get_model_ch4(prior_files)

    ## ---------------------------------------------------------------------##
    ## Set up dask client
    ## ---------------------------------------------------------------------##
    if prior.shape[0] > 4e5:
        n_workers = 1
    else:
        n_workers = 2

    threads_per_worker = 2

    cluster = LocalCluster(n_workers=n_workers,
                           threads_per_worker=threads_per_worker)
    client = Client(cluster)

    ## ---------------------------------------------------------------------##
    ## Iterate through the perturbation directories to build monthly Kw
    ## ---------------------------------------------------------------------##
    # Make a monthly reduced-dimension Jacobian (nobs x npert)
    kw_m = np.array([]).reshape(prior.shape[0], 0)
    for p in perturbation_dirs:
        # Load files
        pert_files = glob.glob(f'{p}/ProcessedDir/{s.year:04d}{month:02d}??_GCtoTROPOMI.pkl')
        pert_files.sort()
        pert = get_model_ch4(pert_files)

        # Get and save the Jacobian column
        diff = (pert - prior).reshape((-1, 1))
        kw_m = np.concatenate((kw_m, diff), axis=1)

    # Convert to xarray
    kw_m = xr.DataArray(kw_m, dims=['nobs', 'nvec'])
    kw_m = kw_m.chunk(chunks={'nobs' : nobs_chunk, 'nvec' : nvec_chunk})

    ## ---------------------------------------------------------------------##
    ## Transform the reduced-dimension Jacobian Kw into state space
    ## ---------------------------------------------------------------------##
    # Calculate the reduced rank jacobian
    k_m = da.tensordot(kw_m, reduction, axes=(1, 0))

    ## ---------------------------------------------------------------------##
    ## Save and exit
    ## ---------------------------------------------------------------------##
    # Save out
    start_time = time.time()
    k_m.to_netcdf(f'{data_dir}k1_m{month:02d}.nc')
    active_time = (time.time() - start_time)/60
    print(f'K for month {month} saved ({active_time} min).')

    # Exit
    print('Code Complete.')
    print('-'*75)
    client.shutdown()
    sys.exit()


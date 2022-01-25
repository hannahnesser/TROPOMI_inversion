if __name__ == '__main__':
    import sys
    import xarray as xr
    # import dask.array as da
    import numpy as np
    import pandas as pd

    import glob
    import time

    ## ---------------------------------------------------------------------##
    ## Set user preferences
    ## ---------------------------------------------------------------------##
    # Cannon
    run_with_script = True
    if run_with_script:
        month = int(sys.argv[1])
        niter = sys.argv[2]
        prior_dir = sys.argv[3]
        perturbation_dirs = sys.argv[4]
        n_perturbation_dirs = int(sys.argv[5])
        data_dir = sys.argv[6]
        code_dir = sys.argv[7]
    else:
        month = 1
        prior_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final'
        perturbation_dirs = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_NNNN'
        n_perturbation_dirs = 434
        data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion'
        code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'

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
    ## Load the observation filter
    ## ---------------------------------------------------------------------##
    if niter == '0':
        obs_filter = pd.read_csv(f'{data_dir}/obs_filter0.csv', header=0)
    else:
        obs_filter = pd.read_csv(f'{data_dir}/obs_filter.csv', header=0)

    obs_filter = obs_filter[obs_filter['MONTH'] == month]['FILTER'].values

    ## ---------------------------------------------------------------------##
    ## Create list of perturbation directories
    ## ---------------------------------------------------------------------##
    perturbation_dirs = [perturbation_dirs.replace('NNNN', f'{i:04d}')
                         for i in range(1, n_perturbation_dirs+1)]
    perturbation_dirs.sort()

    ## ---------------------------------------------------------------------##
    ## Load the data for the prior simulation
    ## ---------------------------------------------------------------------##
    prior_files = glob.glob(f'{prior_dir}/ProcessedDir/{s.year:04d}{month:02d}??_GCtoTROPOMI.pkl')
    prior_files.sort()
    prior = get_model_ch4(prior_files)
    prior = prior[obs_filter]

    ## ---------------------------------------------------------------------##
    ## Set up dask client
    ## ---------------------------------------------------------------------##
    from dask.distributed import Client, LocalCluster, progress
    from dask.diagnostics import ProgressBar
    import dask.config
    import dask.array as da
    dask.config.set({'distributed.comm.timeouts.connect' : 180,
                     'distributed.comm.timeouts.tcp' : 240,
                     'distributed.adaptive.wait-count' : 180,
                     'array.slicing.split_large_chunks' : False,
                     'temporary_directory' : f'{data_dir}/dask-worker-space-{month}'})
    nstate_chunk = 1e3 # int(np.sqrt(max_chunk_size)/5)
    nobs_chunk = 4e4 # int(max_chunk_size/nstate_chunk/5)
    nvec_chunk = len(perturbation_dirs)

    if prior.shape[0] > 4e5:
        n_workers = 1
    else:
        n_workers = 2

    threads_per_worker = 2

    cluster = LocalCluster(n_workers=n_workers,
                           threads_per_worker=threads_per_worker)
    client = Client(cluster)

    ## ---------------------------------------------------------------------##
    ## Load and subset the reduction operator
    ## ---------------------------------------------------------------------##
    reduction = np.load(f'{data_dir}/reduction{(int(niter)-1)}.npy')
    reduction = reduction[:len(perturbation_dirs), :]
    reduction = xr.DataArray(reduction, dims=['nvec', 'nstate'])
    reduction = reduction.chunk(chunks={'nvec' : nvec_chunk,
                                        'nstate' : nstate_chunk})

    ## ---------------------------------------------------------------------##
    ## Iterate through the perturbation directories to build monthly Kw
    ## ---------------------------------------------------------------------##
    # Make a monthly reduced-dimension Jacobian (nobs x npert)
    kw_m = np.array([]).reshape(prior.shape[0], 0)
    for p in perturbation_dirs:
        print(p)

        # Load files
        pert_files = glob.glob(f'{p}/ProcessedDir/{s.year:04d}{month:02d}??_GCtoTROPOMI.pkl')
        pert_files.sort()
        pert = get_model_ch4(pert_files)
        pert = pert[obs_filter]

        # Get and save the Jacobian column
        diff = (pert - prior).reshape((-1, 1))
        kw_m = np.concatenate((kw_m, diff), axis=1)

    # Convert to xarray
    kw_m = xr.DataArray(kw_m, dims=['nobs', 'nvec'])
    kw_m = kw_m.chunk(chunks={'nobs' : nobs_chunk, 'nvec' : nvec_chunk})

    ## ---------------------------------------------------------------------##
    ## Save and exit
    ## ---------------------------------------------------------------------##
    # Transform the reduced-dimension Jacobian Kw into state space
    kpi_m = da.tensordot(kw_m, reduction, axes=(1, 0))
    kpi_m = xr.DataArray(kpi_m, dims=['nobs', 'nstate'])
    kpi_m = kpi_m.chunk({'nobs' : 5e3, 'nstate' : -1})

    # Persist
    kpi_m = kpi_m.persist()
    progress(kpi_m)

    # Save out
    start_time = time.time()
    kpi_m.to_netcdf(f'{data_dir}/k1_m{month:02d}.nc')
    active_time = (time.time() - start_time)/60
    print(f'Kpi for month {month} saved ({active_time} min).')

    # Exit
    print('Code Complete.')
    print('-'*75)
    client.shutdown()
    sys.exit()


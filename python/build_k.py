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
        chunk = int(sys.argv[1])
        chunk_size = int(sys.argv[2])
        niter = sys.argv[3]
        prior_dir = sys.argv[4]
        perturbation_dirs = sys.argv[5]
        n_perturbation_dirs = int(sys.argv[6])
        data_dir = sys.argv[7]
        code_dir = sys.argv[8]
    else:
        chunk = 1
        chunk_size = 125000
        niter = '1'
        prior_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_0000_final'
        perturbation_dirs = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/jacobian_runs/TROPOMI_inversion_NNNN'
        n_perturbation_dirs = 434
        data_dir = f'/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results'
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

    obs_filter = obs_filter['FILTER']
    i = np.cumsum(obs_filter) - 1
    i0 = np.where(i == (chunk-1)*chunk_size)[0][0]
    try:
        i1 = np.where(i == chunk*chunk_size)[0][0]
    except IndexError:
        i1 = obs_filter.shape[0]
    obs_filter[:i0] = False
    obs_filter[i1:] = False

    ## ---------------------------------------------------------------------##
    ## Create list of perturbation directories
    ## ---------------------------------------------------------------------##
    perturbation_dirs = [perturbation_dirs.replace('NNNN', f'{i:04d}')
                         for i in range(1, n_perturbation_dirs+1)]
    perturbation_dirs.sort()

    ## ---------------------------------------------------------------------##
    ## Load the data for the prior simulation
    ## ---------------------------------------------------------------------##
    prior_files = glob.glob(f'{prior_dir}/ProcessedDir/{s.year:04d}????_GCtoTROPOMI.pkl')
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
                     'temporary_directory' : f'{data_dir}/dask-worker-space-{chunk}'})
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
    reduction = np.load(f'{data_dir}/iteration{(int(niter)-1)}/operators/reduction{(int(niter)-1)}.npy')
    reduction = reduction[:len(perturbation_dirs), :]
    reduction = xr.DataArray(reduction, dims=['nvec', 'nstate'])
    reduction = reduction.chunk(chunks={'nvec' : nvec_chunk,
                                        'nstate' : nstate_chunk})

    ## ---------------------------------------------------------------------##
    ## Iterate through the perturbation directories to build a chunk of Kw
    ## ---------------------------------------------------------------------##
    # Make one chunk of the reduced-dimension Jacobian (nobs x npert)
    kw_c = np.array([]).reshape(prior.shape[0], 0)
    for p in perturbation_dirs:
        print(p)

        # Load files
        pert_files = glob.glob(f'{p}/ProcessedDir/{s.year:04d}????_GCtoTROPOMI.pkl')
        pert_files.sort()
        pert = get_model_ch4(pert_files)
        pert = pert[obs_filter]

        # Get and save the Jacobian column
        diff = (pert - prior).reshape((-1, 1))
        kw_c = np.concatenate((kw_c, diff), axis=1)

    # Convert to xarray
    kw_c = xr.DataArray(kw_c, dims=['nobs', 'nvec'])
    kw_c = kw_c.chunk(chunks={'nobs' : nobs_chunk, 'nvec' : nvec_chunk})

    ## ---------------------------------------------------------------------##
    ## Save and exit
    ## ---------------------------------------------------------------------##
    # Transform the reduced-dimension Jacobian Kw into state space
    kpi_c = da.tensordot(kw_c, reduction, axes=(1, 0))
    kpi_c = xr.DataArray(kpi_c, dims=['nobs', 'nstate'])
    kpi_c = kpi_c.chunk({'nobs' : 5e3, 'nstate' : -1})

    # Persist
    kpi_c = kpi_c.persist()
    progress(kpi_c)

    # Save out
    start_time = time.time()
    kpi_c.to_netcdf(f'{data_dir}/iteration{niter}/k/k{niter}_c{chunk:02d}.nc')
    active_time = (time.time() - start_time)/60
    print(f'Kpi for chunk {chunk} saved ({active_time} min).')

    # Exit
    print('Code Complete.')
    print('-'*75)
    client.shutdown()
    sys.exit()


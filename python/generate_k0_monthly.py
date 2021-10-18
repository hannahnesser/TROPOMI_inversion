'''
This python script generates the initial estimate of the Jacobian matrix

Inputs:
    prior_emis      This contains the emissions of the prior run simulation.
                    It is a list of monthly output HEMCO diagnostics files
                    to account for monthly variations in
'''
if __name__ == '__main__':
    import sys
    import time

    import xarray as xr
    import numpy as np
    import pandas as pd

    ## -------------------------------------------------------------------- ##
    ## Set user preferences
    ## -------------------------------------------------------------------- ##
    # Memory constraints
    available_memory_GB = int(sys.argv[1])

    # Data directories
    base_dir = sys.argv[2]
    data_dir = f'{base_dir}/inversion_data/'
    output_dir = sys.argv[3]
    code_dir = sys.argv[4]

    # Import custom packages
    sys.path.append(code_dir)
    import gcpy as gc
    import inversion_settings as s

    # Files
    obs_file = f'{data_dir}/{s.year}_corrected.pkl'
    cluster_file = f'{data_dir}/clusters.nc'
    k_nstate_file = f'{data_dir}/k0_nstate.nc'

    ## -------------------------------------------------------------------- ##
    ## Load the clusters
    ## -------------------------------------------------------------------- ##
    clusters = xr.open_dataarray(cluster_file).squeeze('time')
    nstate = int(clusters.max().values)
    print(f'Number of state vector elements : {nstate}')

    ## -------------------------------------------------------------------- ##
    ## Load and process the observations
    ## -------------------------------------------------------------------- ##
    obs = gc.load_obj(obs_file)[['LON', 'LAT', 'MONTH']]
    nobs = int(obs.shape[0])
    print(f'Number of observations : {nobs}')

    # Subset to reduce memory needs
    obs = obs[['LAT', 'LON', 'MONTH']]
    obs[['MONTH']] = obs[['MONTH']].astype(int)

    ## -------------------------------------------------------------------- ##
    ## Set up a dask client and cacluate the optimal chunk size
    ## -------------------------------------------------------------------- ##
    from dask.distributed import Client, LocalCluster, progress
    from dask.diagnostics import ProgressBar
    import dask.config
    dask.config.set({'distributed.comm.timeouts.connect' : 90,
                     'distributed.comm.timeouts.tcp' : 150,
                     'distributed.adaptive.wait-count' : 90,
                     'array.slicing.split_large_chunks' : False})

    # Open cluster
    n_workers = 3
    threads_per_worker = 2
    cluster = LocalCluster(local_directory=output_dir,
                           n_workers=n_workers,
                           threads_per_worker=threads_per_worker)
    # Open client
    client = Client(cluster)


    # We now calcualte the optimal chunk size for saving the monthly
    # Jacobian. We don't need to do anything but save it, so we use
    # nstate_chunk = nstate and make nobs_chunk as large as possible.
    n_threads = n_workers*threads_per_worker
    max_chunk_size = gc.calculate_chunk_size(available_memory_GB,
                                             n_threads=n_threads)
    nstate_chunk = nstate
    nobs_chunk = int(max_chunk_size/nstate_chunk)
    print('State vector chunks : ', nstate_chunk)
    print('Obs vector chunks   : ', nobs_chunk)

    ## -------------------------------------------------------------------- ##
    ## Generate a monthly K0
    ## -------------------------------------------------------------------- ##
    # Iterate through the months
    for m in s.months:
        print('-'*75)
        print(f'Month {m}')

        # Open k_nstate and select the month
        # Unfortunately, this has to be in the loop because we restart the
        # client each time
        k_nstate = xr.open_dataarray(k_nstate_file,
                                     chunks={'nobs' : nobs_chunk,
                                             'nstate' : nstate_chunk,
                                             'month' : 1})
        k_nstate = k_nstate.sel(month=(m-1)) # obnoxious pythonic indexing

        # Subset obs
        obs_m = obs[obs['MONTH'] == int(m)]
        nobs_m = obs_m.shape[0]
        print(f'In month {m}, there are {nobs_m} observations.')

        # Find the indices that correspond to each observation (i.e. the grid
        # box in which each observation is found) (Yes, this information should
        # be contained in the iGC and jGC columns in obs_file, but stupidly
        # I don't have that information for the cluster files)
        # First, find the cluster number of the grid box of the obs
        lat_idx = gc.nearest_loc(obs_m['LAT'].values, clusters.lat.values)
        lon_idx = gc.nearest_loc(obs_m['LON'].values, clusters.lon.values)
        idx = clusters.values[lat_idx, lon_idx].astype(int)
        print(idx.shape)

        # Subset k_n state
        start_time = time.time()
        k_m = k_nstate[idx, :].chunk({'nobs' : nobs_chunk,
                                      'nstate' : nstate_chunk})
        print(k_m.shape)
        k_m.to_netcdf(f'{output_dir}/k0_m{m:02d}.nc')
        active_time = (time.time() - start_time)/60
        print(f'Month {m} saved ({active_time} min).')

        # Restart the client.
        client.restart()

    # For some reason, the code doesn't exit.
    sys.exit()

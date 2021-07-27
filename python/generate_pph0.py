
if __name__ == '__main__':
    import sys
    import xarray as xr
    import dask.array as da
    import numpy as np
    import pandas as pd

    ## ---------------------------------------------------------------------##
    ## Set user preferences
    ## ---------------------------------------------------------------------##
    # Cannon
    # data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/'
    # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python/'
    data_dir = sys.argv[3]
    code_dir = sys.argv[4]

    # Import custom packages
    import sys
    sys.path.append(code_dir)
    import gcpy as gc
    import inversion_settings as s

    # Month
    # month = 1
    month = int(sys.argv[1])
    print('-'*75)
    print(f'Calculating the prior pre-conditioned Hessian for month {month}')

    # Memory constraints
    available_memory_GB = int(sys.argv[2])

    ## ---------------------------------------------------------------------##
    ## Load pertinent data that defines state and observational dimension
    ## ---------------------------------------------------------------------##
    # Observational error
    so = gc.read_file(f'{data_dir}so.nc')
    nobs_tot = so.shape[0]

    # Prior error
    sa = gc.read_file(f'{data_dir}sa.nc')
    nstate = sa.shape[0]

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

    # Open cluster and client
    n_workers = 1
    threads_per_worker = 2
    cluster = LocalCluster(local_directory=data_dir,
                           n_workers=n_workers,
                           threads_per_worker=threads_per_worker)
    client = Client(cluster)

    # We now calculate chunk size.
    n_threads = n_workers*threads_per_worker
    max_chunk_size = gc.calculate_chunk_size(available_memory_GB,
                                             n_threads=n_threads)
    # We take the squareroot of the max chunk size and scale it down by 5
    # to be safe. It's a bit unclear why this works best in tests.
    nstate_chunk = int(np.sqrt(max_chunk_size)/5)
    nobs_chunk = int(max_chunk_size/nstate_chunk)
    chunks = {'nstate' : nstate_chunk, 'nobs' : nobs_chunk}
    print('State vector chunks : ', nstate_chunk)
    print('Obs vector chunks   : ', nobs_chunk)

    ## ---------------------------------------------------------------------##
    ## Generate the prior pre-conditioned Hessian for that month
    ## ---------------------------------------------------------------------##
    # Get the indices for the month using generic chunks
    i0 = 0
    print(f'{"i0": >22}{"i1": >11}{"n": >11}')
    for m in range(1, month+1):
        k_m = gc.read_file(f'{data_dir}k0_m{m:02d}.nc', chunks=chunks)
        i1 = i0 + k_m.shape[0]
        print(f'Month {m:2d} : {i0:11d}{i1:11d}{(i1-i0):11d}')

        if m != month:
            i0 = i1

    # Subset so
    so_m = so[i0:i1]

    # Calculate the monthly prior pre-conditioned Hessian
    sasqrtkt = k_m*(sa**0.5)
    pph_m = da.tensordot(sasqrtkt.T/so_m, sasqrtkt, axes=(1, 0))
    pph_m = xr.DataArray(pph_m, dims=['nstate_0', 'nstate_1'],
                         name=f'pph0_m{month:02d}')
    print('Prior-pre-conditioned Hessian calculated.')

    # Save out
    pph_m.to_netcdf(f'{data_dir}pph0_m{month:02d}.nc')
    print('Prior-pre-conditioned Hessian saved.')

    # exit
    print('Code Complete.')
    print('-'*75)
    sys.exit()

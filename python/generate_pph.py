if __name__ == '__main__':
    import time
    start_time_global = time.time()

    import sys
    from os import remove
    import glob
    import xarray as xr
    import dask.array as da
    import numpy as np
    import pandas as pd
    import math

    ## ---------------------------------------------------------------------##
    ## Set user preferences
    ## ---------------------------------------------------------------------##
    # Cannon
    # data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion'
    # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
    month = int(sys.argv[1])
    niter = sys.argv[2]
    data_dir = sys.argv[3]
    code_dir = sys.argv[4]

    # Import custom packages
    import sys
    sys.path.append(code_dir)
    import gcpy as gc
    import inversion_settings as s

    # Month
    print('-'*75)
    print(f'Calculating the prior pre-conditioned Hessian for month {month}')

    ## ---------------------------------------------------------------------##
    ## Load pertinent data that defines state and observational dimension
    ## ---------------------------------------------------------------------##
    # Prior error
    sa = gc.read_file(f'{data_dir}/sa.nc')
    nstate = sa.shape[0]

    # Observational suffix
    if niter == '0':
        obs_suffix = '0'
    else:
        obs_suffix = ''

    # Observational error
    so = gc.read_file(f'{data_dir}/so{obs_suffix}.nc')

    # Observations
    y = gc.read_file(f'{data_dir}/y{obs_suffix}.nc')
    ya = gc.read_file(f'{data_dir}/ya{obs_suffix}.nc')
    ydiff = y - ya

    # Observation mask
    obs_filter = pd.read_csv(f'{data_dir}/obs_filter{obs_suffix}.csv', header=0)

    # Get the indices for the month using generic chunks
    i0 = 0
    count = 0
    print(f'{"i0": >22}{"i1": >11}{"n": >11}{"filter":>11}')
    for m in range(1, month+1):
        # Read/subset files
        k_m = gc.read_file(f'{data_dir}/k{niter}_m{m:02d}.nc')
        of_m = obs_filter[obs_filter['MONTH'] == m]['FILTER'].sum()

        # Update indices
        i1 = i0 + k_m.shape[0]

        # Print information
        print(f'Month {m:2d} : {i0:11d}{i1:11d}{(i1-i0):11d}{of_m:11d}')

        count += (i1 - i0)
        if m != month:
            i0 = i1

    # Subset so
    so_m = so[i0:i1]
    ydiff_m = ydiff[i0:i1]
    nobs = so_m.shape[0]

    ## -------------------------------------------------------------------- ##
    ## Set up a dask client and cacluate the optimal chunk size
    ## -------------------------------------------------------------------- ##
    from dask.distributed import Client, LocalCluster, progress
    from dask.diagnostics import ProgressBar
    import dask.config
    dask.config.set({'distributed.comm.timeouts.connect' : 90,
                     'distributed.comm.timeouts.tcp' : 150,
                     'distributed.adaptive.wait-count' : 90,
                     'array.slicing.split_large_chunks' : False,
                     'temporary_directory' : f'{data_dir}/dask-worker-space-{month}'})

    # Open cluster and client
    n_workers = 4
    threads_per_worker = 2

    cluster = LocalCluster(n_workers=n_workers,
                           threads_per_worker=threads_per_worker)
    client = Client(cluster)

    # Set chunk size.
    # We take the squareroot of the max chunk size and scale it down by 5
    # to be safe. It's a bit unclear why this works best in tests.
    nstate_chunk = 1e3 # int(np.sqrt(max_chunk_size)/5)
    nobs_chunk = 4e4 # int(max_chunk_size/nstate_chunk/5)
    chunks = {'nstate' : nstate_chunk, 'nobs' : nobs_chunk}

    ## ---------------------------------------------------------------------##
    ## Generate the prior pre-conditioned Hessian for that month
    ## ---------------------------------------------------------------------##
    # Load k_m
    k_m = gc.read_file(f'{data_dir}/k{niter}_m{month:02d}.nc', chunks=chunks)

    # Initialize our loop
    i = int(0)
    count = 0
    n = 5e4
    print(f'Iterating through {(math.ceil(nobs/n))} chunks.')
    while i <= nobs:
        print('-'*75)
        print(f'Chunk {count}')
        # Subset the Jacobian and observational error
        k_i = k_m[i:(i + int(n)), :]
        so_i = so_m[i:(i + int(n))]
        ydiff_i = ydiff_m[i:(i + int(n))]

        # Calculate the PPH for that subset
        sasqrt_kt_i = k_i*(sa**0.5)
        pph_i = da.tensordot(sasqrt_kt_i.T/so_i, sasqrt_kt_i, axes=(1, 0))
        pph_i = xr.DataArray(pph_i, dims=['nstate_0', 'nstate_1'],
                             name=f'pph{niter}_m{month:02d}')
        pph_i = pph_i.chunk({'nstate_0' : nobs_chunk, 'nstate_1' : nstate})

        # Persist and save
        print('Persisting the prior pre-conditioned Hessian.')
        start_time = time.time()
        pph_i = pph_i.persist()
        progress(pph_i)
        pph_i.to_netcdf(f'{data_dir}/pph{niter}_m{month:02d}_{count:d}.nc')
        active_time = (time.time() - start_time)/60
        print(f'Prior-pre-conditioned Hessian {count} saved ({active_time} min).')

        # Then save out part of what we need for the posterior solution
        pre_xhat_i = da.tensordot(sasqrt_kt_i.T/so_i, ydiff_i, axes=(1, 0))
        pre_xhat_i = xr.DataArray(pre_xhat_i, dims=['nstate'],
                                  name=f'pre_xhat{niter}_m{month:02d}')

        # Persist and save
        print('Persisting the pre-xhat calculation.')
        start_time = time.time()
        pre_xhat_i = pre_xhat_i.persist()
        pre_xhat_m.to_netcdf(f'{data_dir}/pre_xhat{niter}_m{month:02d}_{count:d}.nc')
        active_time = (time.time() - start_time)/60
        print(f'xhat preparation {count} saved ({active_time} min).')

        # Step up
        i = int(i + n)
        count += 1

        # Restart the client
        client.restart()

    # Now sum up the component parts
    print('-'*75)
    client.restart()
    pph_m = xr.DataArray(np.zeros((nstate, nstate)),
                         dims=['nstate_0', 'nstate_1'],
                         name=f'pph{niter}_m{month:02d}')
    pre_xhat_m = xr.DataArray(np.zeros((nstate,)), dims=['nstate'],
                              name=f'pre_xhat{niter}_m{month:02d}')
    for i in range(count):
        print(f'Loading count {i}.')
        temp1 = xr.open_dataarray(f'{data_dir}/pph{niter}_m{month:02d}_{i:d}.nc')
        temp2 = xr.open_dataarray(f'{data_dir}/pre_xhat{niter}_m{month:02d}_{i:d}.nc')
        pph_m += temp1
        pre_xhat_m += temp2

    # Load into memory
    pph_m = pph_m.compute()
    pre_xhat_m = pre_xhat_m.compute().values

    # Save out
    start_time = time.time()
    pph_m.to_netcdf(f'{data_dir}/pph{niter}_m{month:02d}.nc')
    active_time = (time.time() - start_time)/60
    print(f'Prior-pre-conditioned Hessian for month {month} saved ({active_time} min).')

    start_time = time.time()
    pre_xhat_m.to_netcdf(f'{data_dir}/pre_xhat{niter}_m{month:02d}.npy')
    active_time = (time.time() - start_time)/60
    print(f'xhat preparation for month {month} completed ({active_time} min).')

    # Clean up
    files = glob.glob(f'{data_dir}/pph{niter}_m{month:02d}_*.nc')
    files += glob.glob(f'{data_dir}/pre_xhat{niter}_m{month:02d}_*.nc')
    for f in files:
       remove(f)

    # Exit
    print('-'*75)
    active_time_global = (time.time() - start_time_global)/60
    print(f'Code Complete ({active_time_global} min).')
    print('-'*75)
    client.shutdown()
    sys.exit()

    # # Calculate the monthly prior pre-conditioned Hessian
    # sasqrt_kt = k_m*(sa**0.5)
    # pph_m = da.tensordot(sasqrt_kt.T/so_m, sasqrt_kt, axes=(1, 0))
    # pph_m = xr.DataArray(pph_m, dims=['nstate_0', 'nstate_1'],
    #                      name=f'pph{niter}_m{month:02d}')
    # pph_m = pph_m.chunk({'nstate_0' : nobs_chunk, 'nstate_1' : nstate})
    # print('Prior-pre-conditioned Hessian calculated.')

    # # Persist
    # pph_m = pph_m.persist()
    # progress(pph_m)

    # # Save out
    # start_time = time.time()
    # pph_m.to_netcdf(f'{data_dir}/pph{niter}_m{month:02d}.nc')
    # active_time = (time.time() - start_time)/60
    # print(f'Prior-pre-conditioned Hessian for month {month} saved ({active_time} min).')

    # # Then save out part of what we need for the posterior solution
    # sasqrt_kt_soinv_ydiff_m = da.tensordot(sasqrt_kt.T/so_m, ydiff_m,
    #                                        axes=(1, 0))
    # sasqrt_kt_soinv_ydiff_m = xr.DataArray(sasqrt_kt_soinv_ydiff_m,
    #                                        dims=['nstate'],
    #                                        name=f'pre_xhat{niter}_m{month:02d}')

    # # Persist
    # sasqrt_kt_soinv_ydiff_m = sasqrt_kt_soinv_ydiff_m.persist()
    # progress(sasqrt_kt_soinv_ydiff_m)

    # # Save out
    # start_time = time.time()
    # sasqrt_kt_soinv_ydiff_m.to_netcdf(f'{data_dir}/pre_xhat{niter}_m{month:02d}.nc')
    # active_time = (time.time() - start_time)/60
    # print(f'xhat preparation for month {month} completed ({active_time} min).')

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
    # data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results'
    # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
    # niter = '2'
    # chunk = 1
    # chunk_size = 150000
    print(sys.argv)
    chunk = int(sys.argv[1])
    chunk_size = int(sys.argv[2])
    niter = sys.argv[3]
    data_dir = sys.argv[4]
    sa_file = sys.argv[5]
    sa_scale = float(sys.argv[6])
    so_file = sys.argv[7]
    rf = float(sys.argv[8])
    ya_file = sys.argv[9]
    suffix = sys.argv[10]
    code_dir = sys.argv[11]

    if suffix == 'None':
        suffix = ''

    # Import custom packages
    import sys
    sys.path.append(code_dir)
    import gcpy as gc
    import inversion_settings as s

    # chunk
    print('-'*75)
    print(f'Calculating the prior pre-conditioned Hessian for chunk {chunk}')

    ## ---------------------------------------------------------------------##
    ## Load pertinent data that defines state and observational dimension
    ## ---------------------------------------------------------------------##
    # Prior error
    sa = gc.read_file(sa_file)
    sa *= sa_scale**2
    sa_bc = xr.DataArray(0.01**2*np.ones(4), dims=('nstate'))
    nstate = sa.shape[0]

    # Observational suffix
    if niter == '0':
        obs_suffix = '0'
    else:
        obs_suffix = ''

    # Observational error
    so = gc.read_file(so_file)
    so /= rf

    # Observations
    y = gc.read_file(f'{data_dir}/y{obs_suffix}.nc')
    ya = gc.read_file(ya_file)
    ydiff = y - ya

    # Observation mask
    obs_filter = pd.read_csv(f'{data_dir}/obs_filter{obs_suffix}.csv', header=0)

    # Get the indices for the chunk
    i0 = (chunk - 1)*chunk_size
    i1 = chunk*chunk_size

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
                     'temporary_directory' : f'{data_dir}/dask-worker-space-{chunk}'})

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
    nobs_chunk = 5e4 # int(max_chunk_size/nstate_chunk/5)
    chunks = {'nstate' : nstate_chunk, 'nobs' : nobs_chunk}

    ## ---------------------------------------------------------------------##
    ## Generate the prior pre-conditioned Hessian for that chunk
    ## ---------------------------------------------------------------------##
    # Load k_m and k_bc
    k_m = gc.read_file(f'{data_dir}/iteration{niter}/k/k{niter}_c{chunk:02d}.nc',
                       chunks=chunks)
    k_bc = gc.read_file(f'{data_dir}/iteration{niter}/k/k{niter}_bc.nc',
                                chunks=chunks)
    k_bc = k_bc[i0:i1, :]

    # Combine the two Jacobians and add on sa
    k_m = xr.concat([k_m, k_bc], dim='nstate')
    sa = xr.concat([sa, sa_bc], dim='nstate')

    # Initialize our loop
    i = int(0)
    count = 0
    n = 5e4
    print(f'Iterating through {(math.ceil(nobs/n))} chunks.')
    while i < nobs:
        print('-'*75)
        print(f'Chunk {count}')
        # Subset the Jacobian and observational error
        k_i = k_m[i:(i + int(n)), :]
        so_i = so_m[i:(i + int(n))]
        ydiff_i = ydiff_m[i:(i + int(n))]

        # Calculate the PPH for that subset
        k_sasqrt_i = k_i*(sa**0.5)
        pph_i = da.tensordot(k_sasqrt_i.T/so_i, k_sasqrt_i, axes=(1, 0))
        pph_i = xr.DataArray(pph_i, dims=['nstate_0', 'nstate_1'],
                             name=f'pph{niter}_c{chunk:02d}')
        pph_i = pph_i.chunk({'nstate_0' : nobs_chunk, 'nstate_1' : nstate})

        # Persist and save
        print('Persisting the prior pre-conditioned Hessian.')
        start_time = time.time()
        pph_i = pph_i.persist()
        progress(pph_i)
        pph_i.to_netcdf(f'{data_dir}/iteration{niter}/pph/pph{niter}_c{chunk:02d}_{count:d}.nc')
        active_time = (time.time() - start_time)/60
        print(f'Prior-pre-conditioned Hessian {count} saved ({active_time} min).')

        # Then save out part of what we need for the posterior solution
        pre_xhat_i = da.tensordot(k_i.T/so_i, ydiff_i, axes=(1, 0))
        pre_xhat_i = xr.DataArray(pre_xhat_i, dims=['nstate'],
                                  name=f'pre_xhat{niter}_c{chunk:02d}')

        # Persist and save
        print('Persisting the pre-xhat calculation.')
        start_time = time.time()
        pre_xhat_i = pre_xhat_i.persist()
        pre_xhat_i.to_netcdf(f'{data_dir}/iteration{niter}/xhat/pre_xhat{niter}_c{chunk:02d}_{count:d}.nc')
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
    if niter == 2:
        nstate += 4
    pph_m = xr.DataArray(np.zeros((nstate, nstate)),
                         dims=['nstate_0', 'nstate_1'],
                         name=f'pph{niter}_c{chunk:02d}')
    pre_xhat_m = xr.DataArray(np.zeros((nstate,)), dims=['nstate'],
                              name=f'pre_xhat{niter}_c{chunk:02d}')
    for i in range(count):
        print(f'Loading count {i}.')
        temp1 = xr.open_dataarray(f'{data_dir}/iteration{niter}/pph/pph{niter}_c{chunk:02d}_{i:d}.nc')
        temp2 = xr.open_dataarray(f'{data_dir}/iteration{niter}/xhat/pre_xhat{niter}_c{chunk:02d}_{i:d}.nc')
        pph_m += temp1
        pre_xhat_m += temp2

    # Load into memory
    pph_m = pph_m.compute()
    pre_xhat_m = pre_xhat_m.compute()

    # Save out
    start_time = time.time()
    pph_m.to_netcdf(f'{data_dir}/iteration{niter}/pph/pph{niter}{suffix}_c{chunk:02d}.nc')
    active_time = (time.time() - start_time)/60
    print(f'Prior-pre-conditioned Hessian for chunk {chunk} saved ({active_time} min).')

    start_time = time.time()
    pre_xhat_m.to_netcdf(f'{data_dir}/iteration{niter}/xhat/pre_xhat{niter}{suffix}_c{chunk:02d}.nc')
    active_time = (time.time() - start_time)/60
    print(f'xhat preparation for chunk {chunk} completed ({active_time} min).')

    # Clean up
    files = glob.glob(f'{data_dir}/iteration{niter}/pph/pph{niter}_c{chunk:02d}_*.nc')
    files += glob.glob(f'{data_dir}/iteration{niter}/xhat/pre_xhat{niter}_c{chunk:02d}_*.nc')
    for f in files:
       remove(f)

    # Exit
    print('-'*75)
    active_time_global = (time.time() - start_time_global)/60
    print(f'Code Complete ({active_time_global} min).')
    print('-'*75)
    client.shutdown()
    sys.exit()

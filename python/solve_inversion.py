if __name__ == '__main__':
    import sys
    import time
    import copy
    import glob
    from datetime import datetime
    import xarray as xr
    import dask.array as da
    import numpy as np
    import pandas as pd
    from scipy.linalg import eigh
    import matplotlib.pyplot as plt

    ## ---------------------------------------------------------------------##
    ## Set user preferences
    ## ---------------------------------------------------------------------##
    local = False

    # Cannon
    if not local:
        # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
        # data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results'
        # output_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'
        # niter = 2
        niter = sys.argv[1]
        data_dir = sys.argv[2]
        output_dir = sys.argv[3]
        code_dir = sys.argv[4]
        optimize_rf = sys.argv[5]
        rf = float(sys.argv[6])
        sa_in = float(sys.argv[7])
    else:
        niter = 1
        base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
        code_dir = f'{base_dir}python/'
        data_dir = f'{base_dir}inversion_data/'
        optimize_rf = False
        rf = None
        sa_in = None

    # Convert strings to booleans
    if optimize_rf == 'True':
        optimize_rf = True
        print('Calculating cost function.')
    else:
        optimize_rf = False

    # User preferences
    pct_of_info = [50, 70, 75, 80, 90, 99.9]
    snr = None
    rank = None

    ## -------------------------------------------------------------------- ##
    ## Set up working environment
    ## -------------------------------------------------------------------- ##
    # Import custom packages
    import sys
    sys.path.append(code_dir)
    import inversion as inv
    import inversion_settings as s
    import gcpy as gc
    import invpy as ip
    import format_plots as fp
    import config as c

    if not local:
        # Import dask things
        from dask.distributed import Client, LocalCluster, progress
        from dask.diagnostics import ProgressBar
        import dask.config
        dask.config.set({'distributed.comm.timeouts.connect' : 90,
                         'distributed.comm.timeouts.tcp' : 150,
                         'distributed.adaptive.wait-count' : 90,
                         'temporary_directory' : data_dir})

        # Open cluster and client
        n_workers = 2
        threads_per_worker = 2
        cluster = LocalCluster(n_workers=n_workers,
                               threads_per_worker=threads_per_worker)
        client = Client(cluster)

        # We now calculate chunk size.
        n_threads = n_workers*threads_per_worker
        nstate_chunk = 2e6 # orig: 1e3
        nobs_chunk = 5e2 # orig: 5e5
        chunks = {'nstate' : nstate_chunk, 'nobs' : nobs_chunk}

    ## ---------------------------------------------------------------------##
    ## Define functions
    ## ---------------------------------------------------------------------##
    def calculate_Kx(k_dir, x_data, niter=niter, chunks=chunks):
        # List of K files
        k_files = glob.glob(f'{k_dir}/k{niter}_c??.nc')
        k_files.sort()

        # Start time
        start_time = time.time()

        # Iterate
        print('[', end='')
        kx = []
        for i, kf in enumerate(k_files):
            print('-', end='')
            k_n = xr.open_dataarray(kf, chunks=chunks)
            k_n = da.tensordot(k_n, x_data, axes=(1, 0))
            k_n = k_n.compute()
            kx.append(k_n)
        active_time = (time.time() - start_time)/60
        print(f'] {active_time:02f} minutes')

        return np.concatenate(kx)

    def calculate_xhat(evecs, evals_h, pre_xhat, sa):
        # (this formulation only works with constant errors I think)
        xhat = (1 + sa*(evecs*(1/(1+evals_h))) @ evecs.T @ pre_xhat)
        return np.array(xhat)

    def calculate_dofs(evecs, evals_h, sa=None):
        # This formulation definitely only works with constant errors
        evals_q = evals_h/(1 + evals_h)
        a = (evecs*evals_q) @ evecs.T
        return np.diagonal(a)

    def calculate_shat(evecs, evals_h, sa):
        # This formulation only works with diagonal errors
        sa_evecs = evecs*(sa**0.5)
        shat = (sa_evecs*(1/(1 + evals_h))) @ sa_evecs.T
        return np.diagonal(shat)

    ## -------------------------------------------------------------------- ##
    ## Open files
    ## -------------------------------------------------------------------- ##
    # Prior error
    sa = gc.read_file(f'{data_dir}/sa.nc')
    sa = sa.values.reshape(-1, 1)
    nstate = sa.shape[0]

    # Observational error
    so = gc.read_file(f'{data_dir}/so.nc')
    so = so.values.reshape(-1, 1)
    nobs = so.shape[0]

    # Initial pre_xhat information
    pre_xhat = xr.open_dataarray(f'{data_dir}/iteration{niter}/xhat/pre_xhat{niter}.nc')

    # Eigenvectors and eigenvalues
    evecs = np.load(f'{data_dir}/iteration{niter}/operators/evecs{niter}.npy')
    evals_h = np.load(f'{data_dir}/iteration{niter}/operators/evals_h{niter}.npy')

    ## ---------------------------------------------------------------------##
    ## Optimize the regularization factor via cost function analysis
    ## ---------------------------------------------------------------------##
    if optimize_rf:
        # To calculate the cost function for the observational term,
        # we need to load the true observations and the constant term
        ## Load the observations
        y = gc.read_file(f'{data_dir}/y.nc')

        ## Load or solve for the constant value
        try:
            # Try loading the constant value
            c = gc.read_file(f'{data_dir}/c.nc')
        except:
            # If that fails, calculate it
            ## Calculate Kxa
            Kxa = calculate_Kx(f'{data_dir}/iteration{niter}/k',
                               np.ones((nstate,)))

            ## Load the observations generated by the prior simulation
            ya = gc.read_file(f'{data_dir}/ya.nc')

            ## Calculate and save c
            c = ya.values - Kxa
            c = xr.DataArray(c, dims=('nobs'))
            c.to_netcdf(f'{output_dir}/c.nc')

        # We will solve the inversion using 80% of information content
        evals_q_orig = np.load(f'{data_dir}/iteration0/operators/evals_q0.npy')
        rank = ip.get_rank(evals_q=evals_q_orig, pct_of_info=80/100)
        evecs_sub = evecs[:, :rank]
        evals_h_sub = evals_h[:rank]

        # Iterate through different regularization factors and prior
        # errors. Then save out the prior and observational cost function
        rfs = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 2]
        DOFS_threshold = [0.01, 0.05, 0.1]
        sa_i = 1
        # sas = [0.1, 0.25, 0.5, 0.75, 1, 2]
        # sas = [1.5, 2, 3]
        ja = np.zeros((len(rfs), len(DOFS_threshold)))
        # jo = np.zeros((len(rfs), len(DOFS_threshold)))
        n = np.zeros((len(rfs), len(DOFS_threshold)))
        for i, rf_i in enumerate(rfs):
            print(f'Solving the invesion for RF = {rf_i} and Sa = {sa_i}')

            # Scale the relevant terms by RF and Sa
            evals_h_i = sa_i**2*rf_i*copy.deepcopy(evals_h_sub)/0.25
            pre_xhat_i = sa_i*rf_i*copy.deepcopy(pre_xhat)/0.5

            # Calculate the posterior
            xhat = calculate_xhat(evecs_sub, evals_h_i,
                                  pre_xhat_i.values, sa_i)
            dofs = calculate_dofs(evecs_sub, evals_h_i)

            # Subset the posterior
            for j, t_i in enumerate(DOFS_threshold):
                xhat_sub = xhat[dofs >= t_i]
                ja[i, j] = ((xhat_sub - 1)**2/sa_i**2).sum()
                n[i, j] = len(xhat_sub)


            # # # Calculate the posterior observations
            # yhat = calculate_Kx(f'{data_dir}/iteration{niter}/k', xhat)
            # yhat += c.values
            # There's sort of an interesting question here of how
            # to calculate yhat, given that we subset xhat. This will
            # be pertinent when we do our posterior model comparaison

            # # Calculate and save the cost function for the prior term
            # ja[i, j] = ((xhat - np.ones(xhat.shape))**2/1**2).sum()
            # jo[i, j] = ((y.values - yhat)**2/(so.reshape(-1,)/rf_i)).sum()

            # # Calculate the functional state vector size
            # n[i, j] = (dofs >= 0.01).sum()

        # Save the result
        np.save(f'{data_dir}/iteration{niter}/ja{niter}_dofs_threshold.npy', ja)
        # np.save(f'{data_dir}/iteration{niter}/jo{niter}_long_2.npy', jo)
        np.save(f'{data_dir}/iteration{niter}/n_functional_dofs_threshold.npy', n)

    ## ---------------------------------------------------------------------##
    ## Solve the inversion
    ## ---------------------------------------------------------------------##
    # p = 80
    for p in pct_of_info:
        suffix = ''
        if rf is not None:
            suffix = suffix + f'_rf{rf}'
        if sa_in is not None:
            suffix = suffix + f'_sa{(sa_in)}'

        print(f'Using {p} percent of information content.')
        evals_q_orig = np.load(f'{data_dir}/iteration0/operators/evals_q0.npy')
        rank = ip.get_rank(evals_q=evals_q_orig, pct_of_info=p/100)
        suffix = suffix + f'_poi{p}'

        # If RF is defined, scale
        if rf is not None:
            evals_h *= rf
            pre_xhat *= rf

        # If sa_in is defined, scale
        if sa_in is not None:
            evals_h *= sa_in**2/0.25
            pre_xhat *= sa_in/0.5
            sa *= sa_in**2/0.25

        # Recompute evals_q.
        evals_q = evals_h/(1 + evals_h)

        # Subset the evals and evecs
        evals_h_sub = evals_h[:rank]
        evals_q_sub = evals_q[:rank]
        evecs_sub = evecs[:, :rank]

        # Calculate the posterior and averaging kernel
        # (we can leave off Sa when it's constant)
        # xhat = (np.sqrt(sa)*evecs_sub/(1+evals_q_sub)) @ evecs_sub.T
        # a = (evecs_sub*evals_q_sub) @ evecs_sub.T
        dofs = calculate_dofs(evecs_sub, evals_h_sub)
        xhat = calculate_xhat(evecs_sub, evals_h_sub, pre_xhat.values, sa)
        shat = calculate_shat(evecs_sub, evals_h_sub, sa)

        # Save the result
        # np.save(f'{data_dir}/iteration{niter}/a/a{niter}{suffix}.npy', a)
        np.save(f'{data_dir}/iteration{niter}/a/dofs{niter}{suffix}.npy', dofs)
        np.save(f'{data_dir}/iteration{niter}/xhat/xhat{niter}{suffix}.npy', xhat)
        np.save(f'{data_dir}/iteration{niter}/xhat/shat{niter}{suffix}.npy', shat)

        print('CODE COMPLETE')

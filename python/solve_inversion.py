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
        # output dir is currently unused
        # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
        # data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results'
        # output_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'
        # niter = '2'
        # ya_file = f'{data_dir}/ya.nc'
        # c_file = f'{data_dir}/c.nc'
        # so_file = f'{data_dir}/so_rg4rt.nc'
        # sa_file = f'{data_dir}/sa.nc'
        # sa_scale = 1
        # rf = 1
        niter = sys.argv[1]
        data_dir = sys.argv[2]
        output_dir = sys.argv[3]
        optimize_bc = sys.argv[4]
        optimize_rf = sys.argv[5]
        xa_abs_file = sys.argv[6]
        sa_file = sys.argv[7]
        sa_scale = float(sys.argv[8])
        so_file = sys.argv[9]
        rf = float(sys.argv[10])
        ya_file = sys.argv[11]
        c_file = sys.argv[12]
        suffix = sys.argv[13]
        code_dir = sys.argv[14]

    else:
        niter = 1
        base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
        code_dir = f'{base_dir}python/'
        data_dir = f'{base_dir}inversion_data/'
        optimize_rf = False
        rf = None
        sa_in = None

    if suffix == 'None':
        suffix = ''

    # Convert strings to booleans
    if optimize_bc == 'True':
        optimize_bc = True
        suffix = '_bc' + suffix
        print('Optimizing boundary condition elements.')
    else:
        optimize_bc = False

    if optimize_rf == 'True':
        optimize_rf = True
        print('Calculating cost function.')
    else:
        optimize_rf = False

    # User preferences
    # pct_of_info = [50, 70, 75, 80, 90, 99.9]
    pct_of_info = 80

    ## -------------------------------------------------------------------- ##
    ## Set up working environment
    ## -------------------------------------------------------------------- ##
    # Import custom packages
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
                         'temporary_directory' : f'{data_dir}/inv_dask_worker{suffix}'})

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

        # if niter == 2, also load the boundary condition K
        if optimize_bc:
            k_bc = xr.open_dataarray(f'{k_dir}/k{niter}_bc.nc',
                                     chunks=chunks)

        # Start time
        start_time = time.time()

        # Iterate
        print('[', end='')
        kx = []
        i0 = 0
        for i, kf in enumerate(k_files):
            print('-', end='')
            k_n = xr.open_dataarray(kf, chunks=chunks)
            # Append the BC K if it's the second iteration
            if optimize_bc:
                i1 = i0 + k_n.shape[0]
                k_bc_n = k_bc[i0:i1, :]
                k_n = xr.concat([k_n, k_bc_n], dim='nstate')
                i0 = copy.deepcopy(i1)
            k_n = da.tensordot(k_n, x_data, axes=(1, 0))
            k_n = k_n.compute()
            kx.append(k_n)
        active_time = (time.time() - start_time)/60
        print(f'] {active_time:02f} minutes')

        return np.concatenate(kx)

    def calculate_xhat(shat, kt_so_ydiff):
        # (this formulation only works with constant errors I think)
        return np.array(1 + shat @ kt_so_ydiff)

    def calculate_dofs(evecs, evals_h, sa):
        evals_q = evals_h/(1 + evals_h)
        a = sa**0.5*(evecs*evals_q) @ evecs.T*(1/(sa**0.5))
        return a

    def calculate_shat(evecs, evals_h, sa):
        # This formulation only works with diagonal errors
        sa_evecs = evecs*(sa**0.5)
        shat = (sa_evecs*(1/(1 + evals_h))) @ sa_evecs.T
        return shat

    def solve_inversion(evecs, evals_h, sa, kt_so_ydiff):
        shat = calculate_shat(evecs, evals_h, sa)
        xhat = calculate_xhat(shat, kt_so_ydiff)
        dofs = calculate_dofs(evecs, evals_h, sa)
        return xhat, shat, dofs

    ## -------------------------------------------------------------------- ##
    ## Open files
    ## -------------------------------------------------------------------- ##
    # Ratio of standard prior to input prior
    xa_abs_base = gc.read_file(f'{data_dir}/xa_abs.nc')
    xa_abs = gc.read_file(xa_abs_file)
    xa_ratio = xa_abs/xa_abs_base
    xa_ratio[xa_abs == 0] = 1 # Correct for the one grid cell with 0 emisisons
    xa_ratio = xa_ratio.values.reshape((-1,))

    # Prior error
    sa = gc.read_file(sa_file)
    sa = sa.values.reshape(-1, 1)

    # If niter == 2, add in BC
    if optimize_bc:
        sa = np.concatenate([sa, 0.01**2*np.ones((4, 1))])

    # Get the state vector dimension
    nstate = sa.shape[0]

    # Observational error (apply RF later)
    so = gc.read_file(so_file)
    so = so.values.reshape(-1, 1)
    nobs = so.shape[0]

    # Load or solve for the constant value
    try:
        # Try loading the constant value
        c = gc.read_file(c_file)
    except:
        # If that fails, calculate it
        ## Calculate Kxa
        Kxa = calculate_Kx(f'{data_dir}/iteration{niter}/k', xa_ratio)

        ## Load the observations generated by the prior simulation
        ya = gc.read_file(ya_file)

        ## Calculate and save c
        c = ya.values - Kxa
        c = xr.DataArray(c, dims=('nobs'))
        c.to_netcdf(c_file)

    # Initial pre_xhat information
    pre_xhat = xr.open_dataarray(f'{data_dir}/iteration{niter}/xhat/pre_xhat{niter}{suffix}.nc').values

    # Eigenvectors and eigenvalues
    evecs = np.load(f'{data_dir}/iteration{niter}/operators/evecs{niter}{suffix}.npy')
    evals_h = np.load(f'{data_dir}/iteration{niter}/operators/evals_h{niter}{suffix}.npy')

    # If sa_scale is defined, scale everything
    if sa_scale is not None:
        evals_h *= sa_scale**2
        sa *= sa_scale**2

    # Subset the eigenvectors and eigenvalues
    print(f'Using {pct_of_info} percent of information content.')
    evals_q_orig = np.load(f'{data_dir}/iteration0/operators/evals_q0.npy')
    rank = ip.get_rank(evals_q=evals_q_orig, pct_of_info=pct_of_info/100)

    # Subset the evals and evecs
    evals_h = evals_h[:rank]
    evecs = evecs[:, :rank]

    ## ---------------------------------------------------------------------##
    ## Optimize the regularization factor via cost function analysis
    ## ---------------------------------------------------------------------##
    if optimize_rf:
        # THIS MIGHT BE WRONG AFTER SWITCHING PASSAGE OF SA AND SO
        # To calculate the cost function for the observational term,
        # we need to load the true observations and the constant term
        ## Load the observations
        y = gc.read_file(f'{data_dir}/y.nc')

        # Iterate through different regularization factors and prior
        # errors. Then save out the prior and observational cost function
        rfs = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 2]
        # DOFS_threshold = [0.01, 0.05, 0.1]
        DOFS_threshold = 0.05
        # sas = [0.1, 0.25, 0.5, 0.75, 1, 2]
        # sas = [1.5, 2, 3]
        ja = np.zeros((len(rfs)))#, len(DOFS_threshold)))
        jo = np.zeros((len(rfs)))#, len(DOFS_threshold)))
        n = np.zeros((len(rfs)))#, len(DOFS_threshold)))
        for i, rf_i in enumerate(rfs):
            print(f'Solving the invesion for RF = {rf_i} and Sa = {sa[0]}')

            # Scale the relevant terms by RF and Sa
            evals_h_i = rf_i*copy.deepcopy(evals_h)
            p_i = rf_i*copy.deepcopy(pre_xhat)

            # Calculate the posterior
            xhat, _, dofs = solve_inversion(evecs, evals_h_i, sa, p_i)

            # # Calculate the posterior observations
            yhat = calculate_Kx(f'{data_dir}/iteration{niter}/k', xhat)
            yhat += c.values
            # There's sort of an interesting question here of how
            # to calculate yhat, given that we subset xhat. This will
            # be pertinent when we do our posterior model comparaison

            # Save out
            suff = suffix + f'_rf{rf_i}' + f'_sax{sa_scale}' + f'_poi{pct_of_info}'
            np.save(f'{data_dir}/iteration{niter}/a/dofs{niter}{suff}.npy', dofs)
            np.save(f'{data_dir}/iteration{niter}/xhat/xhat{niter}{suff}.npy', xhat)
            np.save(f'{data_dir}/iteration{niter}/y/y{niter}{suff}.npy', yhat)

            # Subset the posterior
            # for j, t_i in enumerate(DOFS_threshold):
            xhat[dofs < DOFS_threshold] = 1 #t_i]

            # Calculate and save the cost function for the prior term
            ja[i] = ((xhat - 1)**2/sa.reshape(-1,)).sum()
            jo[i] = ((y.values - yhat)**2/(so.reshape(-1,)/rf_i)).sum()
            n[i] = (dofs >= DOFS_threshold).sum()

            # # Calculate the functional state vector size
            # n[i, j] = (dofs >= 0.01).sum()

        # Save the result
        np.save(f'{data_dir}/iteration{niter}/ja{niter}.npy', ja)
        np.save(f'{data_dir}/iteration{niter}/jo{niter}.npy', jo)
        np.save(f'{data_dir}/iteration{niter}/n_functional{niter}.npy', n)

    ## ---------------------------------------------------------------------##
    ## Solve the inversion
    ## ---------------------------------------------------------------------##
    # set the suffix
    if rf is not None:
        suffix = suffix + f'_rf{rf}'
    if sa_scale is not None:
        suffix = suffix + f'_sax{sa_scale}'
    suffix = suffix + f'_poi{pct_of_info}'

    # If RF is defined, scale
    if rf is not None:
        evals_h *= rf
        pre_xhat *= rf
        so /= rf

    # Recompute evals_q.
    # evals_q = evals_h/(1 + evals_h)
    # evals_q_sub = evals_q[:rank]

    # Calculate the posterior and averaging kernel
    # (we can leave off Sa when it's constant)
    # xhat = (np.sqrt(sa)*evecs_sub/(1+evals_q_sub)) @ evecs_sub.T
    # a = (evecs_sub*evals_q_sub) @ evecs_sub.T
    xhat, shat, a = solve_inversion(evecs, evals_h, sa, pre_xhat)
    dofs = np.diagonal(a)

    # Save the result
    # np.save(f'{data_dir}/iteration{niter}/a/a{niter}{suffix}.npy', a)
    np.save(f'{data_dir}/iteration{niter}/a/dofs{niter}{suffix}.npy', dofs)
    np.save(f'{data_dir}/iteration{niter}/xhat/xhat{niter}{suffix}.npy', xhat)
    np.save(f'{data_dir}/iteration{niter}/shat/shat{niter}{suffix}.npy', shat)

    # # Calculate the posterior observations
    # yhat = calculate_Kx(f'{data_dir}/iteration{niter}/k', xhat)
    # yhat += c.values

    # Save the result
    # np.save(f'{data_dir}/iteration{niter}/y/y{niter}{suffix}.npy', yhat)

    print('CODE COMPLETE')
    print(f'Saved xhat{niter}{suffix}.nc and more.')

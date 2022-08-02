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
        # output_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data'
        # niter = '2'
        # xa_abs_file = f'{data_dir}/xa_abs_wetlands404_edf_bc0.nc'
        # ya_file = f'{data_dir}/ya.nc'
        # c_file = f'{data_dir}/c_wetlands404_edf_bc0.nc'
        # so_file = f'{data_dir}/so_rg2rt_10t.nc'
        # sa_file = f'{data_dir}/sa.nc'
        # sa_scale = 1
        # rf = 1
        # suffix = '_rg2rt_10t_wetlands404_edf_bc0'
        # pct_of_info = 80
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
        pct_of_info = float(sys.argv[13])
        suffix = sys.argv[14]
        code_dir = sys.argv[15]

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
    # pct_of_info = 80

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

    ## -------------------------------------------------------------------- ##
    ## Open files
    ## -------------------------------------------------------------------- ##
    # Prior error
    sa = gc.read_file(sa_file)
    sa = sa.values.reshape(-1, 1)

    # If niter == 2, add in BC
    if optimize_bc:
        sa = np.concatenate([sa, 0.01**2*np.ones((4, 1))])

    # Get the state vector dimension
    nstate = sa.shape[0]

    # # Observational error (apply RF later)
    # so = gc.read_file(so_file)
    # so = so.values.reshape(-1, 1)
    # nobs = so.shape[0]

    # # Observations
    # y = gc.read_file(f'{data_dir}/y{obs_suffix}.nc')
    # ya = gc.read_file(ya_file)

    # # Update ya for new prior
    # # This part should be deleted eventually in preference of using
    # # the correct ya from GEOS-Chem
    # if (xa_abs_file.split('/')[-1] != 'xa_abs_correct.nc'):
    #     xa_abs = gc.read_file(xa_abs_file).reshape(-1,)
    #     xa_abs_orig = gc.read_file(f'{data_dir}/xa_abs_correct.nc').reshape(-1,)
    #     xa_ratio = xa_abs/xa_abs_orig
    #     Kxd = ip.calculate_Kx(f'{data_dir}/iteration{niter}/k', xa_ratio - 1)
    #     ya += Kxd
    #     print(f'Updated modeled observations yield maximum {ya.max()} and minimum {ya.min()}')
    # else:
    #     xa_ratio = np.ones(nstate)

    # # Calculate ydiff
    # ydiff = y - ya

    # Initial pre_xhat information
    pre_xhat = xr.open_dataarray(f'{data_dir}/iteration{niter}/xhat/pre_xhat{niter}{suffix}.nc').values

    # Eigenvectors and eigenvalues
    evecs = np.load(f'{data_dir}/iteration{niter}/operators/evecs{niter}{suffix}.npy')
    evals_h = np.load(f'{data_dir}/iteration{niter}/operators/evals_h{niter}{suffix}.npy')

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
        # y = gc.read_file(f'{data_dir}/y.nc')

        # Iterate through different regularization factors and prior
        # errors. Then save out the prior and observational cost function
        rfs = [0.01, 0.05, 0.1, 0.5, 1.0]
        sas = [0.5, 0.75, 1.0, 1.25, 1.5]
        dds = [0.05, 0.1, 0.15, 0.2]
        ja = np.zeros((len(rfs), len(sas), len(dds)))
        n = np.zeros((len(rfs), len(sas), len(dds)))
        negs = np.zeros((len(rfs), len(sas), len(dds)))
        avg = np.zeros((len(rfs), len(sas), len(dds)))
        for i, rf_i in enumerate(rfs):
            for j, sa_i in enumerate(sas):
                print(f'Solving the invesion for RF = {rf_i} and Sa = {sa_i}')

                # Scale the relevant terms by RF and Sa
                evals_h_ij = rf_i*sa_i**2*copy.deepcopy(evals_h)
                p_ij = rf_i*copy.deepcopy(pre_xhat)

                if optimize_bc:
                    sa_ij = copy.deepcopy(sa)
                    sa_ij[:-4] = sa_ij[:-4]*sa_i**2
                else:
                    sa_ij = copy.deepcopy(sa)*sa_i**2

                # Calculate the posterior
                _, xh_fr, _, a = ip.solve_inversion(evecs, evals_h_ij, sa_ij, p_ij)
                dofs = np.diagonal(a)

                # # Calculate the posterior observations
                # yhat = calculate_Kx(f'{data_dir}/iteration{niter}/k', xhat)
                # yhat += c.values
                # # There's sort of an interesting question here of how
                # # to calculate yhat, given that we subset xhat. This will
                # # be pertinent when we do our posterior model comparaison

                # Save out
                suff = suffix + f'_rf{rf_i}' + f'_sax{sa_i}' + f'_poi{pct_of_info}'
                np.save(f'{data_dir}/iteration{niter}/a/dofs{niter}{suff}.npy', dofs)
                np.save(f'{data_dir}/iteration{niter}/xhat/xhat_fr{niter}{suff}.npy', xh_fr)
                # np.save(f'{data_dir}/iteration{niter}/y/y{niter}{suff}.npy', yhat)

                # Subset the posterior
                # for j, t_i in enumerate(DOFS_threshold):
                for k, dofs_i in enumerate(dds):
                    xh_fr[dofs < dofs_i] = 1
                    nf = (dofs >= dofs_i).sum()

                    # Calculate and save the cost function for the prior term
                    ja[i, j, k] = ((xh_fr - 1)**2/sa_ij.reshape(-1,)).sum()/nf
                    n[i, j, k] = nf
                    negs[i, j, k] = (xh_fr < 0).sum()
                    avg[i, j, k] = xh_fr[dofs >= dofs_i].mean()

        # Save the result
        np.save(f'{data_dir}/iteration{niter}/ja{niter}{suffix}.npy', ja)
        np.save(f'{data_dir}/iteration{niter}/n_func{niter}{suffix}.npy', n)
        np.save(f'{data_dir}/iteration{niter}/negs{niter}{suffix}.npy', negs)
        np.save(f'{data_dir}/iteration{niter}/avg{niter}{suffix}.npy', avg)

        # np.save(f'{data_dir}/iteration{niter}/jo{niter}{suffix}.npy', jo)

    ## ---------------------------------------------------------------------##
    ## Solve the inversion
    ## ---------------------------------------------------------------------##
    # set the suffix and scale
    if rf is not None:
        suffix = suffix + f'_rf{rf}'
        evals_h *= rf
        pre_xhat *= rf
        so /= rf

    if sa_scale is not None:
        suffix = suffix + f'_sax{sa_scale}'
        evals_h *= sa_scale**2
        if optimize_bc:
            sa[:-4] = sa[:-4]*sa_scale**2
        else:
            sa *= sa_scale**2

    # Update suffix for pct of info
    suffix = suffix + f'_poi{pct_of_info}'

    # Recompute evals_q.
    # evals_q = evals_h/(1 + evals_h)
    # evals_q_sub = evals_q[:rank]

    # Calculate the posterior and averaging kernel
    # (we can leave off Sa when it's constant)
    # xhat = (np.sqrt(sa)*evecs_sub/(1+evals_q_sub)) @ evecs_sub.T
    # a = (evecs_sub*evals_q_sub) @ evecs_sub.T
    xhat, xhat_fr, shat, a = ip.solve_inversion(evecs, evals_h, sa, pre_xhat)
    dofs = np.diagonal(a)

    # Save the result
    # np.save(f'{data_dir}/iteration{niter}/a/a{niter}{suffix}.npy', a)
    np.save(f'{data_dir}/iteration{niter}/a/dofs{niter}{suffix}.npy', dofs)
    np.save(f'{data_dir}/iteration{niter}/xhat/xhat{niter}{suffix}.npy', xhat)
    np.save(f'{data_dir}/iteration{niter}/xhat/xhat_fr{niter}{suffix}.npy', 
            xhat_fr)
    np.save(f'{data_dir}/iteration{niter}/shat/shat{niter}{suffix}.npy', shat)

    # # Calculate the posterior observations
    # yhat = calculate_Kx(f'{data_dir}/iteration{niter}/k', xhat)
    # yhat += c.values

    # Save the result
    # np.save(f'{data_dir}/iteration{niter}/y/y{niter}{suffix}.npy', yhat)

    print('CODE COMPLETE')
    print(f'Saved xhat{niter}{suffix}.nc and more.')

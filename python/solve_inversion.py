if __name__ == '__main__':
    import sys
    import copy
    import xarray as xr
    import dask.array as da
    import numpy as np

    ## ---------------------------------------------------------------------##
    ## Set user preferences
    ## ---------------------------------------------------------------------##
    # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python'
    # data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results'
    # niter = '2'
    # xa_abs_file = f'{data_dir}/xa_abs_wetlands404_edf_bc0.nc'
    # ya_file = f'{data_dir}/ya.nc'
    # so_file = f'{data_dir}/so_rg2rt_10t.nc'
    # sa_file = f'{data_dir}/sa.nc'
    # sa_scale = 1
    # rf = 1
    # suffix = '_bc_rg2rt_10t_wetlands404_edf_bc0'
    # pct_of_info = 80
    # optimize_bc = True
    niter = sys.argv[1]
    data_dir = sys.argv[2]
    optimize_bc = sys.argv[3]
    optimize_rf = sys.argv[4]
    xa_abs_file = sys.argv[5]
    sa_file = sys.argv[6]
    sa_scale = float(sys.argv[7])
    so_file = sys.argv[8]
    rf = float(sys.argv[9])
    ya_file = sys.argv[10]
    pct_of_info = float(sys.argv[11])
    evec_sf = float(sys.argv[12]) 
    suffix = sys.argv[13]
    code_dir = sys.argv[14]

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

    ## -------------------------------------------------------------------- ##
    ## Set up working environment
    ## -------------------------------------------------------------------- ##
    # Import custom packages
    sys.path.append(code_dir)
    import gcpy as gc
    import invpy as ip

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

    # Scale by the eigenvector scaling
    evals_h *= 1/evec_sf**2
    pre_xhat *= 1/evec_sf

    ## ---------------------------------------------------------------------##
    ## Optimize the regularization factor via cost function analysis
    ## ---------------------------------------------------------------------##
    if optimize_rf:
        # Iterate through different regularization factors and prior
        # errors. Then save out the prior and observational cost function.
        rfs = [0.01, 0.05, 0.1, 0.5, 1.0]
        sas = [0.5, 1.0, 2.0, 3.0, 4.0]
        dds = [0.05, 0.1, 0.15, 0.2]
        ja_fr = np.zeros((len(rfs), len(sas), len(dds)))
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
                sa_ij = copy.deepcopy(sa)*sa_i**2

                # Calculate the posterior
                xh, xh_fr, _, a = ip.solve_inversion(evecs, evals_h_ij, 
                                                    sa_ij, p_ij)
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
                np.save(f'{data_dir}/iteration{niter}/xhat/xhat{niter}{suff}.npy', xh)
                np.save(f'{data_dir}/iteration{niter}/xhat/xhat_fr{niter}{suff}.npy', xh_fr)
                # np.save(f'{data_dir}/iteration{niter}/y/y{niter}{suff}.npy', yhat)

                # Subset for boundary condition
                if optimize_bc:
                    xh_fr = xh_fr[:-4]
                    dofs = dofs[:-4]

                # Subset the posterior
                # for j, t_i in enumerate(DOFS_threshold):
                for k, dofs_i in enumerate(dds):
                    xh_fr[dofs < dofs_i] = 1
                    nf = (dofs >= dofs_i).sum()

                    # Calculate and save the cost function for the prior term
                    ja_fr[i, j, k] = ((xh_fr - 1)**2/sa_ij.reshape(-1,)).sum()/nf
                    ja[i, j, k] = ((xh - 1)**2/sa_ij.reshape(-1,)).sum()/nf
                    n[i, j, k] = nf
                    negs[i, j, k] = (xh_fr < 0).sum()
                    avg[i, j, k] = xh_fr[dofs >= dofs_i].mean()

        # Save the result
        np.save(f'{data_dir}/iteration{niter}/ja{niter}{suffix}.npy', ja)
        np.save(f'{data_dir}/iteration{niter}/ja_fr{niter}{suffix}.npy', ja_fr)
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

    if sa_scale is not None:
        suffix = suffix + f'_sax{sa_scale}'
        evals_h *= sa_scale**2
        if optimize_bc:
            sa[:-4] = sa[:-4]*sa_scale**2
        else:
            sa *= sa_scale**2

    # Update suffix for pct of info
    suffix = suffix + f'_poi{pct_of_info}'

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

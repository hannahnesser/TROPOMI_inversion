if __name__ == '__main__':
    import sys
    from datetime import datetime
    import xarray as xr
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
        # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python/'
        # data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/'
        # output_dir = '/n/seasasfs02/hnesser/TROPOMI_inversion/inversion_data/'
        niter = sys.argv[1]
        n_evecs = int(sys.argv[2])
        data_dir = sys.argv[3]
        output_dir = sys.argv[4]
        code_dir = sys.argv[5]
        calculate_evecs = bool(sys.argv[6])
        format_evecs = bool(sys.argv[7])
        solve_inversion = bool(sys.argv[8])
    else:
        niter = 1
        n_evecs = int(10)
        base_dir = '/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/'
        code_dir = f'{base_dir}python/'
        data_dir = f'{base_dir}inversion_data/'
        calculate_evecs = False
        format_evecs = False
        solve_inversion = False

    # User preferences
    pct_of_info = [50, 80, 90, 95, 99, 99.9]
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
        nstate_chunk = 1e3
        chunks = {'nstate_0' : nstate_chunk, 'nstate_1' : nstate_chunk}
        print('State vector chunks : ', nstate_chunk)

    ## -------------------------------------------------------------------- ##
    ## Load global quantities
    ## -------------------------------------------------------------------- ##
    # State vector dimension
    sa = gc.read_file(f'{data_dir}/sa.nc')
    sa = sa.values.reshape(-1, 1)
    nstate = sa.shape[0]

    ## ---------------------------------------------------------------------##
    ## Calculate the eigenvectors
    ## ---------------------------------------------------------------------##
    if calculate_evecs:
        # Sum together the monthly PPHs and the pre-xhat calculation
        pph = xr.DataArray(np.zeros((nstate, nstate)),
                           dims=['nstate_0', 'nstate_1'], name=f'pph{niter}')
        pre_xhat = xr.DataArray(np.zeros((nstate,)), dims=['nstate'],
                                name=f'pre_xhat{niter}')
        for m in s.months:
            print(f'Loading month {m}.')
            temp1 = xr.open_dataarray(f'{data_dir}/pph{niter}_m{m:02d}.nc')
            temp2 = xr.open_dataarray(f'{data_dir}/pre_xhat{niter}_m{m:02d}.nc')
            # print(m, temp1.min(), temp1.max())
            pph += temp1
            pre_xhat += temp2

        # Load into memory
        pph = pph.compute()
        pre_xhat = pre_xhat.compute()

        # Calculate the eigenvectors (this is the time consuming step)
        evals_h, evecs = eigh(pph)

        # Sort them
        idx = np.argsort(evals_h)[::-1]
        evals_h = evals_h[idx]
        evecs = evecs[:, idx]

        # Force all evals to be non-negative
        if (evals_h < 0).sum() > 0:
            print('Negative eigenvalues. Maximum negative value is %.2e. Setting negative eigenvalues to zero.' \
                % (evals_h[evals_h < 0].min()))
            evals_h[evals_h < 0] = 0

        # Check for imaginary eigenvector components and force all
        # eigenvectors to be only the real component.
        if np.any(np.iscomplex(evecs)):
            print('Imaginary eigenvectors exist at index %d of %d. Forcing eigenvectors to real component alone.' \
                  % ((np.where(np.iscomplex(evecs))[1][0] - 1), len(evecs)))
            evecs = np.real(evecs)

        # Calculate evals_q
        evals_q = evals_h/(1 + evals_h)

        # Calculate the prolongation and reduction operators
        prolongation = (sa**0.5) * evecs
        reduction = evecs.T * (1/sa**0.5)

        # Save out the matrices
        pph.to_netcdf(f'{data_dir}/pph{niter}.nc')
        pre_xhat.to_netcdf(f'{data_dir}/pre_xhat{niter}.nc')
        np.save(f'{data_dir}/evecs{niter}.npy', evecs)
        np.save(f'{data_dir}/evals_h{niter}.npy', evals_h)
        np.save(f'{data_dir}/evals_q{niter}.npy', evals_q)
        np.save(f'{data_dir}/prolongation{niter}.npy', prolongation)
        np.save(f'{data_dir}/reduction{niter}.npy', reduction)

        print('Eigendecomposition complete.\n')

    else:
        pre_xhat = xr.open_dataarray(f'{data_dir}/pre_xhat{niter}.nc')
        evals_h = np.load(f'{data_dir}/evals_h{niter}.npy')
        evecs = np.load(f'{data_dir}/evecs{niter}.npy')
        prolongation = np.load(f'{data_dir}/prolongation{niter}.npy')
        if not local:
            evals_q = np.load(f'{data_dir}/evals_q{niter}.npy')
            reduction = np.load(f'{data_dir}/reduction{niter}.npy')
        else:
            evals_q = evals_h/(1 + evals_h)

    ## ---------------------------------------------------------------------##
    ## Format the eigenvectors for HEMCO
    ## ---------------------------------------------------------------------##
    if format_evecs:
        # First, print information about the rank/percent of information content/
        # SNR associated with the number of evecs provided
        DOFS_frac = np.cumsum(evals_q)/evals_q.sum()
        print(f'SAVING OUT {n_evecs} EIGENVECTORS')
        print(f'% DOFS : {(100*DOFS_frac[n_evecs])}')
        print(f'SNR    : {(evals_h[n_evecs]**0.5)}')

        # Load clusters
        clusters = xr.open_dataarray(f'{data_dir}/clusters.nc')

        # Iterate through columns and save out HEMCO-appropriate files
        for i in range(n_evecs):
            pert = ip.match_data_to_clusters(prolongation[:, i], clusters, 0)
            # pert = ip.match_data_to_clusters(prolongation[:, i], clusters, 0)

            # Define HEMCO attributes
            long_name = 'Eigenvector perturbations'
            title_str = f'Eigenvector perturbation {i+1} for the construction of the Jacobian matrix for methane inversions.'
            pert = gc.define_HEMCO_std_attributes(pert, name='evec_pert')
            pert = gc.define_HEMCO_var_attributes(pert, 'evec_pert',
                                                  long_name=long_name,
                                                  units='kg/m2/s')
            pert.attrs = {'Title' : title_str,
                          'Conventions' : 'COARDS',
                          'History' : datetime.now().strftime('%Y-%m-%d %H:%M')}

            # Scaling (for a later date)
            # for s in evec_scaling:
            #     suffix = f'_{(i+1):04d}_{s}'
            #     pert.attrs = {'Title' : title_str, 'Scaling' : s}
            #     p = deepcopy(pert)
            #     p['evec_pert'] *= float(s)

            gc.save_HEMCO_netcdf(pert, f'{output_dir}/inversion_data/eigenvectors{niter}', f'evec_pert_{(i+1):04d}.nc')
            print(f'Saved eigenvector {(i+1)} : {output_dir}/inversion_data/eigenvectors{niter}/evec_pert_{(i+1):04d}.nc')

    ## ---------------------------------------------------------------------##
    ## Solve the inversion
    ## ---------------------------------------------------------------------##
    if solve_inversion:
        print('Calculating averaging kernel.')
        for p in pct_of_info:
            print(f'Using {p} percent of information content.')
            # Figure out the fraction of information content
            # if sum(x is not None for x in [p, snr, rank]) > 1:
            #     raise AttributeError('Conflicting rank arguments provided.')
            # elif sum(x is not None for x in [p, snr, rank]) == 0:
            #     raise AttributeError('Insufficient rank arguments provided.')
            # elif p is not None:
            evals_q_orig = np.load(f'{data_dir}/evals_q0.npy')
            rank = ip.get_rank(evals_q=evals_q_orig, pct_of_info=p/100)
            # diff = np.abs(DOFS_frac - (p/100))
            # rank = np.argwhere(diff == np.min(diff))[0][0]
            suffix = f'_poi{p}'
            # elif snr is not None:
            #     evals_h[evals_h < 0] = 0
            #     diff = np.abs(evals_h**0.5 - snr)
            #     rank = np.argwhere(diff == np.min(diff))[0][0]
            #     suffix = f'_snr{snr}'
            # else:
            #     suffix = f'_rank{rank}'
            # print(f'Rank = {rank}')

            # Subset the evals and evecs
            evals_h_sub = evals_h[:rank]
            evals_q_sub = evals_q[:rank]
            evecs_sub = evecs[:, :rank]

            # Calculate the posterior and averaging kernel
            # (we can leave off Sa when it's constant)
            # xhat = (np.sqrt(sa)*evecs_sub/(1+evals_q_sub)) @ evecs_sub.T
            a = (evecs_sub*evals_q_sub) @ evecs_sub.T
            xhat = (np.ones(pre_xhat.values.shape) +
                    (evecs_sub*sa*(1/(1+evals_h_sub))) @ evecs_sub.T @ pre_xhat.values)

            # Save the result
            np.save(f'{data_dir}/a{niter}{suffix}.npy', a)
            np.save(f'{data_dir}/dofs{niter}{suffix}.npy', np.diagonal(a))
            np.save(f'{data_dir}/xhat{niter}{suffix}.npy', xhat)

    print('CODE COMPLETE')

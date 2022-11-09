if __name__ == '__main__':
    import sys
    from os import remove
    import glob
    from datetime import datetime
    import xarray as xr
    import numpy as np
    import pandas as pd
    from scipy.linalg import eigh

    ## ---------------------------------------------------------------------##
    ## Set user preferences
    ## ---------------------------------------------------------------------##
    # niter
    # code_dir = '/n/home04/hnesser/TROPOMI_inversion/python/'
    # data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results/'
    # output_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data/'
    # suffix = '_bc_rg2rt_10t_w404_edf_bc0'
    # suffix2 = '_rg2rt_10t_w404_edf_nlc'
    niter = sys.argv[1]
    n_evecs = int(sys.argv[2])
    data_dir = sys.argv[3]
    output_dir = sys.argv[4]
    optimize_bc = sys.argv[5]
    calculate_evecs = sys.argv[6]
    format_evecs = sys.argv[7]
    sa_file = sys.argv[8]
    sa_scale = float(sys.argv[9])
    suffix = sys.argv[10]
    code_dir = sys.argv[11]

    if suffix == 'None':
        suffix = ''

    # Convert strings to booleans
    if optimize_bc == 'True':
        optimize_bc = True
        suffix = '_bc' + suffix
        print('Optimizing boundary condition elements.')
    else:
        optimize_bc = False

    if calculate_evecs == 'True':
        calculate_evecs = True
        print('Calculating eigenvectors.')
    else:
        calculate_evecs = False

    if format_evecs == 'True':
        format_evecs = True
        print('Formatting eigenvectors.')
    else:
        format_evecs = False

    # User preferences
    pct_of_info = [50, 80, 90, 99.9]
    snr = None
    rank = None

    print(f'Solving inversion for {suffix}')

    ## -------------------------------------------------------------------- ##
    ## Set up working environment
    ## -------------------------------------------------------------------- ##
    # Import custom packages
    sys.path.append(code_dir)
    import inversion as inv
    import inversion_settings as s
    import gcpy as gc
    import invpy as ip
    import config as c

    # Import dask things
    from dask.distributed import Client, LocalCluster, progress
    from dask.diagnostics import ProgressBar
    import dask.config
    dask.config.set({'distributed.comm.timeouts.connect' : 90,
                     'distributed.comm.timeouts.tcp' : 150,
                     'distributed.adaptive.wait-count' : 90,
                     'temporary_directory' : f'{data_dir}/evecs_dask_worker{suffix}'})

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
    sa = gc.read_file(sa_file)
    sa = sa.values.reshape(-1, 1)
    if optimize_bc:
        sa = np.concatenate([sa, 10**2*np.ones((4, 1))])
    sa *= sa_scale**2

    # Get the state vector dimension
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
        for c in range(1, 21):
            print(f'Loading chunk {c}.')
            temp1 = xr.open_dataarray(f'{data_dir}/iteration{niter}/pph/pph{niter}{suffix}_c{c:02d}.nc')
            temp2 = xr.open_dataarray(f'{data_dir}/iteration{niter}/xhat/pre_xhat{niter}{suffix}_c{c:02d}.nc')
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
        pph.to_netcdf(f'{data_dir}/iteration{niter}/pph/pph{niter}{suffix}.nc')
        pre_xhat.to_netcdf(f'{data_dir}/iteration{niter}/xhat/pre_xhat{niter}{suffix}.nc')
        np.save(f'{data_dir}/iteration{niter}/operators/evecs{niter}{suffix}.npy', evecs)
        np.save(f'{data_dir}/iteration{niter}/operators/evals_h{niter}{suffix}.npy', evals_h)
        np.save(f'{data_dir}/iteration{niter}/operators/evals_q{niter}{suffix}.npy', evals_q)
        np.save(f'{data_dir}/iteration{niter}/operators/prolongation{niter}{suffix}.npy', prolongation)
        np.save(f'{data_dir}/iteration{niter}/operators/reduction{niter}{suffix}.npy', reduction)

        # Clean up
        files = glob.glob(f'{data_dir}/iteration{niter}/pph/pph{niter}{suffix}_c*.nc')
        files += glob.glob(f'{data_dir}/iteration{niter}/xhat/pre_xhat{niter}{suffix}_c*.nc')
        for f in files:
           remove(f)

        print('Eigendecomposition complete.\n')
        print(f'Saved evecs{niter}{suffix}.nc and more.')

    else:
        evals_h = np.load(f'{data_dir}/iteration{niter}/operators/evals_h{niter}{suffix}.npy')
        # evals_q = evals_h/(1 + evals_h)
        prolongation = np.load(f'{data_dir}/iteration{niter}/operators/prolongation{niter}{suffix}.npy')
        if not local:
            evals_q = np.load(f'{data_dir}/iteration{niter}/operators/evals_q{niter}{suffix}.npy')

    ## ---------------------------------------------------------------------##
    ## Format the eigenvectors for HEMCO
    ## ---------------------------------------------------------------------##
    if format_evecs:
        # First, print information about the rank/percent of information
        # content/SNR associated with the number of evecs provided
        DOFS_frac = np.cumsum(evals_q)/evals_q.sum()
        print(f'SAVING OUT {n_evecs} EIGENVECTORS')
        print(f'% DOFS : {(100*DOFS_frac[n_evecs])}')
        print(f'SNR    : {(evals_h[n_evecs]**0.5)}')

        # Load clusters
        clusters = xr.open_dataarray(f'{data_dir}/clusters.nc')

        # Iterate through columns and save out HEMCO-appropriate files
        for i in range(n_evecs):
            pert = ip.match_data_to_clusters(prolongation[:, i], clusters, 0)

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

            gc.save_HEMCO_netcdf(pert, f'{output_dir}/inversion_data/eigenvectors{niter}', f'evec_pert_{(i+1):04d}.nc')
            print(f'Saved eigenvector {(i+1)} : {output_dir}/inversion_data/eigenvectors{niter}/evec_pert_{(i+1):04d}.nc')

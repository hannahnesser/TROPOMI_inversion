if __name__ == '__main__':

    from dask.distributed import Client, LocalCluster, progress
    cluster = LocalCluster(n_workers=6, threads_per_worker=2)
    client = Client(cluster)

    import xarray as xr
    import dask.array as da
    import sys
    sys.path.append('/n/home04/hnesser/TROPOMI_inversion/python')
    import inversion as inv

    # Load data
    data_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/initial_inversion/'
    k   = [f'{data_dir}k0_m{i:02d}.nc' for i in range(1, 13)]
    xa  = f'{data_dir}xa.nc'
    sa  = f'{data_dir}sa.nc'
    y   = f'{data_dir}y.nc'
    ya  = f'{data_dir}ya.nc'
    so  = f'{data_dir}so.nc'
    c   = f'{data_dir}c.nc'

    # Create inversion object
    data = inv.Inversion(k, xa, sa, y, ya, so, c,
                         regularization_factor=1, reduced_memory=True,
                         available_memory_GB=45, k_is_positive=True)

    # Deal with annoying things
    data.so = data.so.drop('nobs')
    data.so = data.so.rename('so')
    data.sa = data.sa.rename('sa')

    # Load Sa and So into memory... not sure if this is the right thing
    # to do
    data.sa = data.sa.compute()
    data.so = data.so.compute()

    #
    nstate_chunk = 1e2
    nobs_chunk = -1

    pph_temp = (data.sa**0.5)*data.k.T
    pph_temp = pph_temp.chunk((nstate_chunk, nobs_chunk))
    # data.so = data.so.chunk(nobs_chunk)
    pph = da.einsum('ij,jk', pph_temp, pph_temp.T/data.so)
    # pph = pph.rechunk((1e4, 1e3))

    pph = client.persist(pph)
    progress(pph)

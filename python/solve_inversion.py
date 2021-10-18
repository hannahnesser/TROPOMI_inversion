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
    run_with_script = False
    if run_with_script:

    else:

    ## ---------------------------------------------------------------------##
    ## Start dask cluster and client
    ## ---------------------------------------------------------------------##
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

    # Set chunk size
    n_state_chunk = 1e3



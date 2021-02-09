'''
This is a package designed to deal with common TROPOMI
processing needs.
'''

import numpy as np
import xarray as xr

def blended_albedo_filter(data, data_swir, data_nir):
    '''
    Returns a filter that is true where we keep the data and false
    elsewhere.
    '''
    filt = (2.4*data_nir - 1.13*data_swir)
    #print('The filter preserves %f%% of data.' % (100*filt.sum()/len(filt)))
    return filt



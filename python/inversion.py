import xarray as xr
from dask.diagnostics import ProgressBar
import dask.array as da
import numpy as np
# from numpy import diag as diags
# from numpy import identity
from numpy.linalg import inv, norm #, eigh
from scipy.sparse import diags, identity
from scipy.stats import linregress
from scipy.linalg import eigh
import pandas as pd
import copy
import warnings

# clustering
from sklearn.cluster import KMeans

import math

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams, colorbar, colors
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom
import cartopy.crs as ccrs
import cartopy

# Import information for plotting in a consistent fashion
import config
import format_plots as fp
import invpy as ip
import gcpy as gc

# Other font details
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'AppleGothic'
rcParams['font.size'] = config.LABEL_FONTSIZE*config.SCALE
rcParams['text.usetex'] = True
# rcParams['mathtext.fontset'] = 'stixsans'
rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'
rcParams['axes.titlepad'] = 0

'''
This class creates an inversion object that contains the quantities
 necessary for conducting an analytic inversion. It also defines the
following functions:
    calculate_c             Calculate the constant in y = Kx + c
    obs_mod_diff            Calculate the difference between modeled
                            observations and observations (y - Kx)
    cost_func               Calculate the cost function for a given x
    solve_inversion         Solve the analytic inversion, inclduing the
                            posterior emissions, error, and information
                            content
It also defines the following plotting functions:
    plot_state              Plot a state vector on an emissions grid
    plot_state_grid         Plot multiple state vectors on multiple
                            emissions grids
    plot_multiscale_grid    Plot the grid for a reduced dimension,
                            multiscale emissions state vector
'''

class Inversion:
### todo: this should really be set in each class that needs it
    # Set global variables
    dims = {'xa' : ['nstate'], 'sa' : ['nstate', 'nstate'],
            'y' : ['nobs'], 'ya' : ['nobs'], 'so' : ['nobs', 'nobs'],
            'c' : ['nobs'], 'k' : ['nobs', 'nstate']}
    def __init__(self, k, xa, sa, y, ya, so, c=None, regularization_factor=1,
                 k_is_positive=False):
        '''
        Define an inversion object with the following required
        inputs:
            k               The Jacobian matrix for the forward model,
                            K = dy/dx, which represents the sensitivity of the
                            simulated observations to each element of the
                            state vector
            xa              The prior for the state vector
            sa              The prior error variances as a vector. This class
                            is not currently written to accept error
                            covariances (such a change would be simple to
                            implement).
            y               The observations
            ya              The simulated observations generated from the prior
                            state vector
            so              The observational error variances as a vector,
                            including errors from the forward model and the
                            observations. This class is not currently written
                            to accept error covariances, though such a change
                            would be simple to implement
        All inputs can either be arrays or arrays. Any arrays passed to the
        class will be coerced into xarrays.

        The object also accepts the following inputs:
            c                        An array or file name containing a vector
                                     of dimension nobs that represents the
                                     intercept of the forward model:
                                            y = F(x) + c
            regularization_factor    A regularization factor gamma defined in
                                     the cost function as
                            J(x) = (x-xa)T Sa (x-xa) + gamma(y-Kx)T So (y-Kx).
                                     The regularization factor changes the
                                     weighting of the observational term
                                     relative to the prior term by reducing So
                                     by a factor of gamma.
            k_is_positive            A flag indicating whether the Jacobian
                                     has already been checked for negative
                                     elements. Set to true if you have already
                                     checked for negative elements, or if you 
                                     want to keep negative elements (e.g. OH). 
                                     Default is False.

        The object then defines the following:
            nstate          The dimension of the state vector
            nobs            The dimension of the observation vector
            xhat            Empty, held for the posterior state vector
            shat            Empty, held for the posterior error
            a               Empty, held for the averaging kernel, a measure
                            of the information content of the inverse system
            yhat            Empty, the updated simulated observations, defined
                            as y = Kxhat + c
        '''
        print('... Initializing inversion object ...')

        # Read in the elements of the inversion
        self.regularization_factor = regularization_factor

        # Read in the prior elements first because that allows us to calculate
        # nstate and thus the chunk size in that dimension
        print('Loading the prior and prior error.')
        self.xa = self.read(xa, dims=self.dims['xa'])
        self.sa = self.read(sa, dims=self.dims['sa'])

        # Save out the state vector dimension
        self.nstate = self.xa.shape[0]
        print(f'State vector dimension : {self.nstate}\n')

        # Load observational data
        print('Loading the Jacobian, observations, and observational error.')
        self.k = self.read(k, dims=self.dims['k'])
        self.y = self.read(y, dims=self.dims['y'])
        self.ya = self.read(ya, dims=self.dims['ya'])
        self.so = self.read(so, dims=self.dims['so'])

        # Save out the observation vector dimension
        self.nobs = self.y.shape[0]
        print(f'Observation vector dimension : {self.nobs}\n')

        # Modify so by the regularization factor
        if self.regularization_factor != 1:
            self.so /= self.regularization_factor

        # Check that the dimensions match up
        self._check_dimensions()

        # Force k to be positive
        if not k_is_positive:
            print('Checking the Jacobian for negative values.')
            if (self.k < 0).any():
                print('Forcing negative values of the Jacobian to 0.\n')
                self.k = np.where(self.k > 0, self.k, 0)

        # Solve for the constant c.
        if c is None:
            print('Calculating c as c = y - F(xa) = y - ya')
            self.c = self.calculate_c()
        else:
            self.c = self.read(c, dims=self.dims['c'], chunks=chunks)

        # Now create some holding spaces for values that may be filled
        # in the course of solving the inversion.
        self.xhat = None
        self.shat = None
        self.a = None
        self.dofs = None # diagonal elements of A
        self.yhat = None

        print('... Complete ...\n')

    #########################
    ### UTILITY FUNCTIONS ###
    #########################
    # todo: likely will want to move this to the big_mem version
    def _get_dims(self,sa,so):
        # Get dimensions for an xarray dataset 
        dims = {'xa' : ['nstate'], 'sa' : ['nstate', 'nstate'],
                'y' : ['nobs'], 'ya' : ['nobs'], 'so' : ['nobs', 'nobs'],
                'c' : ['nobs'], 'k' : ['nobs', 'nstate']}
        # check if prior error is a 1D vector
        # todo: this will break if you have a list of filenames :/ but self.read currently breaks on this condition so I'll fix it if we decide to fix that.
        if type(sa) == str: 
            pass # filenames don't need dims b/c they are read to datasets
        elif len(sa.shape) == 1:
            dims['sa'] = ['nstate']
        # check if observation error is a 1D vector
        if type(so) == str: 
            pass # filenames don't need dims b/c they are read to datasets
        elif len(so.shape) == 1:
            dims['so'] = ['nobs']
        return dims

    def read(self, item, dims=None, **kwargs):
        # If item is a string or a list, load the file
        if (type(item) == str) or (type(item) == list):
            item = np.array(gc.read_file(*[item], **kwargs))        # Force the items to np.arrays
        # Force the items to be dataarrays
        elif type(item) != np.ndarray:
            item = np.array(item)
        return item

    @staticmethod
    def _to_dataarray(item, dims):
        # If it's a dataset, require that there be only one variable
        if type(item) == xr.core.dataset.Dataset:
            variables = list(item.keys())
            assert len(variables) == 1, \
                  'A dataset with multiple variables was provided. This \
                   class requires that any datasets contain only one \
                   variable.'
            item = item[variables[0]]
        # Else convert to a dataarray
        else:
            assert dims is not None, \
                   'Creating an xarray dataset and dims is not provided.'
            item = xr.DataArray(item, dims=tuple(dims))
        return item

    def save(self, attribute:str, file_name=None):
        item = getattr(self, attribute)

        # Get file name
        if file_name is None:
            file_name = attribute

        item.to_netcdf(f'{file_name}.nc')

    def add_attr_to_dataset(self, ds:xr.Dataset, attr_str:str, dims:tuple):
        '''
        Adds an attribute from Inversion instance to an xarray Dataset. Modiefies Dataset inplace.
        '''
        attr = getattr(self, attr_str).squeeze()
        if attr is None:
            print(f'{attr_str} has not been assigned yet - skipping.')
        else:
            if dims is None: # for 0-d attributes
                ds = ds.assign_attrs({attr_str : attr})
            else:
                ds[attr_str] = xr.DataArray(data=attr, dims=Inversion.dims)

    def to_dataset(self):
        '''
        Convert an instance of the Inversion object to an xarray Dataset.
        Vectors are stored as DataArrays, and scalars are stored
        as attributes in the Dataset.
        '''
        # warn that there is no custom implementation for each subclass
        if type(self)!=Inversion:
            warnings.warn('Inversion.to_xarray will only export attributes present in the Inversion class. Specialized attributes from subclasses (e.g. ReducedRankInversion) are not currently exported to the dataset, so some attributes might be missing. If you would like to add individual attributes to your dataset, you can use the add_attrs_to_dataset method.', stacklevel=2)

        # intialize dataset
        ds = xr.Dataset()

        # store vectors as DataArrays
        # inputs
        self.add_attr_to_dataset(ds, 'state_vector', ('nstate'))
        self.add_attr_to_dataset(ds, 'k',            ('nobs','nstate'))
        self.add_attr_to_dataset(ds, 'xa',           ('nstate'))
        self.add_attr_to_dataset(ds, 'sa',       ('nstate'))
        self.add_attr_to_dataset(ds, 'y',            ('nobs'))
        self.add_attr_to_dataset(ds, 'ya',       ('nobs'))
        self.add_attr_to_dataset(ds, 'so',       ('nobs'))
        # outputs
        self.add_attr_to_dataset(ds, 'c',     ('nobs'))
        self.add_attr_to_dataset(ds, 'xhat',  ('nstate'))
        self.add_attr_to_dataset(ds, 'shat',  ('nstate','nstate'))
        self.add_attr_to_dataset(ds, 'a',     ('nstate','nstate'))
        self.add_attr_to_dataset(ds, 'y_out', ('nobs'))
        self.add_attr_to_dataset(ds, 'dofs',  ('nstate'))
        # scalar values (stored as attributes, dims=None)
        self.add_attr_to_dataset(ds, 'rf',     None)

        return ds

    def _check_dimensions(self):
        # Check whether the items in the object match the dimensions


        # Check whether all inputs have the right dimensions
        assert self.k.shape[1] == self.nstate, \
               'Dimension mismatch: Jacobian and prior.'
        assert self.k.shape[0] == self.nobs, \
               'Dimension mismatch: Jacobian and observations.'
        assert self.so.shape[0] == self.nobs, \
               'Dimension mismatch: observational error'
        assert self.sa.shape[0] == self.nstate, \
               'Dimension mismatch: prior error.'

    @staticmethod
    def _find_dimension(data):
        if len(data.shape) == 1:
            return 1
        elif data.shape[1] == 1:
            return 1
        else:
            return 2


    ####################################
    ### STANDARD INVERSION FUNCTIONS ###
    ####################################
    def calculate_c(self):
        '''
        Calculate c for the forward model, defined as ybase = Kxa + c.
        Save c as an element of the object.
        '''
        c = self.ya - self.k @ self.xa
        return c

    def obs_mod_diff(self, x):
        '''
        Calculate the difference between the true observations y and the
        simulated observations Kx + c for a given x. It returns this
        difference as a vector.

        Parameters:
            x      The state vector at which to evaluate the difference
                   between the true and simulated observations
        Returns:
            diff   The difference between the true and simulated observations
                   as a vector
        '''
        diff = self.y - (self.k @ x + self.c)
        return diff

    def cost_func(self, x):
        '''
        Calculate the value of the Bayesian cost function
            J(x) = (x - xa)T Sa (x-xa) + regularization_factor(y - Kx)T So (y - Kx)
        for a given x. Prints out that value and the contributions from
        the emission and observational terms.

        Parameters:
            x      The state vector at which to evaluate the cost function
        Returns:
            cost   The value of the cost function at x
        '''

        # Calculate the observational component of the cost function
        cost_obs = self.obs_mod_diff(x).T \
                   @ diags(1/self.so) @ self.obs_mod_diff(x)

        # Calculate the emissions/prior component of the cost function
        cost_emi = (x - self.xa).T @ diags(1/self.sa) @ (x - self.xa)

        # Calculate the total cost, print out information on the cost, and
        # return the total cost function value
        cost = cost_obs + cost_emi
        print('     Cost function: %.2f (Emissions: %.2f, Observations: %.2f)'
              % (cost, cost_emi, cost_obs))
        return cost

    def solve_inversion(self):
        '''
        Calculate the solution to an analytic Bayesian inversion for the
        given Inversion object. The solution includes the posterior state
        vector (xhat), the posterior error covariance matrix (shat), and
        the averaging kernel (A). The function prints out progress statements
        and information about the posterior solution, including the value
        of the cost function at the prior and posterior, the number of
        negative state vector elements in the posterior solution, and the
        DOFS of the posterior solution.
        '''
        print('... Solving inversion ...')

        # Check if the errors are diagonal or not and, if so, modify
        # calculations involving the error covariance matrices
        if (self._find_dimension(self.sa) == 1):
            sainv = diags(1/self.sa)
        else:
            sainv = inv(self.sa)

        if (self._find_dimension(self.so) == 1):
            kTsoinv = (self.k.T/self.so)
        else:
            kTsoinv = self.k.T @ inv(self.so)

        # Reshape y, ya, xa, and c to include the trailing 1
        # this is required to play nice with sparse matrix
        self.y  = self.y.reshape(-1,1)
        self.ya = self.ya.reshape(-1,1)
        self.xa = self.xa.reshape(-1,1)
        self.c  = self.c.reshape(-1,1)
        
        # Calculate the cost function at the prior.
        print('Calculating the cost function at the prior mean.')
        cost_prior = self.cost_func(self.xa)

        # Calculate the posterior error.
        print('Calculating the posterior error.')
        self.shat = inv(kTsoinv @ self.k + sainv)

        # Calculate the posterior mean
        print('Calculating the posterior mean.')
        #gain = self.shat @ kTsoinv
        #self.xhat = self.xa + (gain @ self.obs_mod_diff(self.xa))
        self.xhat = self.xa + (self.shat @ kTsoinv @ self.obs_mod_diff(self.xa))

        # Calculate the cost function at the posterior. Also calculate the
        # number of negative cells as an indicator of inversion success.
        print('Calculating the cost function at the posterior mean.')
        cost_post = self.cost_func(self.xhat)
        print('     Negative cells: %d' % self.xhat[self.xhat < 0].sum())

        # Calculate the averaging kernel.
        print('Calculating the averaging kernel.')
        self.a = identity(self.nstate) - self.shat @ sainv
        self.dofs = np.diag(self.a)
        print('     DOFS: %.2f' % np.trace(self.a))

        # Calculate the new set of modeled observations.
        print('Calculating updated modeled observations.')
        self.yhat = self.k @ self.xhat + self.c

        print('... Complete ...\n')

    ##########################
    ### PLOTTING FUNCTIONS ###
    ##########################

    def plot_state(self, attribute, clusters_plot, default_value=0,
                   cbar=True, **kw):
        # Get the data from the attribute argument. The attribute argument
        # is either a string or a string and int. ## EXPAND COMMENT HERE
        try:
            attribute, selection = attribute
            data = getattr(self, attribute)[:, selection]
            attribute_str = attribute + '_' + str(selection)
        except ValueError:
            data = getattr(self, attribute)
            attribute_str = attribute

        kw['title'] = kw.get('title', attribute_str)

        # Match the data to lat/lon data
        fig, ax, c = ip.plot_state(np.array(data).squeeze(), clusters_plot,
                                   default_value, cbar, **kw)
        return fig, ax, c

    def plot_state_grid(self, attributes, rows, cols, clusters_plot,
                        cbar=True, **kw):
        assert rows*cols == len(attributes), \
               'Dimension mismatch: Data does not match number of plots.'

        try:
            kw.get('vmin')
            kw.get('vmax')
        except KeyError:
            print('vmin and vmax not supplied. Plots may have inconsistent\
                   colorbars.')

        try:
            titles = kw.pop('titles')
            vmins = kw.pop('vmins')
            vmaxs = kw.pop('vmaxs')
        except KeyError:
            pass

        fig_kwargs = kw.pop('fig_kwargs', {})
        fig, ax = fp.get_figax(rows, cols, maps=True,
                               lats=clusters_plot.lat, lons=clusters_plot.lon,
                                **fig_kwargs)

        if cbar:
            cax = fp.add_cax(fig, ax)
            cbar_kwargs = kw.pop('cbar_kwargs', {})

        for i, axis in enumerate(ax.flatten()):
            kw['fig_kwargs'] = {'figax' : [fig, axis]}
            try:
                kw['title'] = titles[i]
                kw['vmin'] = vmins[i]
                kw['vmax'] = vmaxs[i]
            except NameError:
                pass

            fig, axis, c = self.plot_state(attributes[i], clusters_plot,
                                           cbar=False, **kw)
        if cbar:
            cbar_title = cbar_kwargs.pop('title', '')
            c = fig.colorbar(c, cax=cax, **cbar_kwargs)
            c = fp.format_cbar(c, cbar_title)

        return fig, ax, c

class InversionLoop(Inversion):
    """
    TODO - errors MUST be diagonal.
    """
    def __init__(self, k, xa, sa, y, ya, so, c=None, regularization_factor=1,
                 k_is_positive=False):
        super.__init__(self, k, xa, sa, y, ya, so, c=None, regularization_factor=1,
                 k_is_positive=False)
        # check that k is a list of filenames
        assert (type(self.k) == list), ('k is not a list. '
                                        'k must be a list of nstate filenames, '
                                        'where each file is a column of k')
        assert (len(self.k) == self.nstate), (f'k has {len(self.k)} elements but should have {self.nstate}. '
                                              'k must be a list of nstate filenames, '
                                              'where each file is a column of k')
    
    # Slightly different read function for InversionLoop 
    # Keeps lists unread. Instead they are read while you solve to conserve memory.
    def read(self, item, dims=None, **kwargs):
        # If item is a list, leave it (we will loop over it later)
        if type(item) == list:
            pass
        # If item is a string, load the file
        elif type(item) == str:
            item = np.array(gc.read_file(*[item], **kwargs))        
        # Force the items to np.arrays
        elif type(item) != np.ndarray:
            item = np.array(item)
        return item    

    # Code taken Lu Shen & Daniel Varon's TROPOMI_Inversion script
    def solve_inversion(self):
    # ==========================================================================================
    # Now we will assemble different terms of the analytical inversion.
    #
    # These are the terms of eq. (5) and (6) in Zhang et al. (2018) ACP:
    # "Monitoring global OH concentrations using satellite observations of atmospheric methane".
    #
    # Specifically, we are going to solve:
    #   xhat = xA + G*(y-K*xA) 
    #        = xA + inv(gamma*K^T*inv(S_o)*K + inv(S_a)) * gamma*K^T*inv(S_o) * (y-K*xA)
    #
    # In the code below this becomes
    #   xhat = xA + inv(gamma*kT_invso + inv(S_a)) * gamma*kT_invso_ydiff
    #        = xA + ratio
    #        = 1  + ratio      [since xA=1 when optimizing scale factors]
    # ========================================================================================== 
        # filenames are stored in self.k for InversionLoop only 
        k_fname_list = self.k 
        # Initialize two parts of the inversion equation
        all_part1 = np.zeros([n_clust,n_clust], dtype=float)
        all_part2 = np.zeros([n_clust], dtype=float)

        # For each .nc file generated by generate_k0.py (todo is this the file?):
        for fname in k_fname_list:

            # Load TROPOMI/GEOS-Chem and Jacobian matrix data from the .nc file
            dat = read(fname)

            # Otherwise, grab the TROPOMI/GEOS-Chem data
            obs_GC = dat['obs_GC']
            
            # Only consider data within the new latitude and longitude bounds
            ind = np.where((obs_GC[:,2]>=xlim[0]) & (obs_GC[:,2]<=xlim[1]) & 
                           (obs_GC[:,3]>=ylim[0]) & (obs_GC[:,3]<=ylim[1]))[0]

            # Skip if no data in bounds
            if (len(ind) == 0):
                continue

            # TROPOMI and GEOS-Chem data within bounds
            obs_GC = obs_GC[ind,:]

            # Jacobian entries for observations within bounds [ppb]
            K = 1e9 * dat['KK'][ind,:]

            # Number of observations
            N = obs_GC.shape[0]
            print('Sum of Jacobian entries:', np.sum(K))

            # Define observational errors (diagonal entries of S_o matrix)
            obs_error = np.zeros(N)
            obs_error.fill(obs_err**2)
        
            # Measurement-model mismatch: TROPOMI columns minus GEOS-Chem virtual TROPOMI columns
            # This is (y - F(xA)), or (y - (K*xA + O)), or (y - K*xA) in shorthand
            deltaY = obs_GC[:,0] - obs_GC[:,1] # [ppb]
            
            # If there are any nan's in the data, abort 
            if (np.any(np.isnan(deltaY)) or np.any(np.isnan(K)) or np.any(np.isnan(obs_error))):
                print('missing values', fi)
                break
        
            # Define KT_invSo = K^T*inv(S_o)
            KT = K.transpose() 
            KT_invSo = np.zeros(KT.shape, dtype=float)
            for k in range(KT.shape[1]):
                KT_invSo[:,k] = KT[:,k]/obs_error[k]        

            # Parts of inversion equation
            part1 = KT_invSo@K             # K^T*inv(S_o)*K
            part2 = KT_invSo@deltaY        # K^T*inv(S_o)*(y-K*xA)
       
            # Add part1 & part2 to sums 
            all_part1 += part1
            all_part2 += part2
                
            # Inverse of prior error covariance matrix, inv(S_a)
            emis_error = np.zeros(n_clust)
            emis_error.fill(prior_err**2)    
            inv_Sa = np.diag(1/emis_error)   # Inverse of prior error covariance matrix

            # Solve for posterior scaling factors, xhat
            ratio = np.linalg.inv(gamma*all_part1 + inv_Sa)@(gamma*all_part2)
            xhat = 1 + ratio

            return xhat, ratio, all_part1, all_part2

class ReducedRankInversion(Inversion):
    def __init__(self, k, xa, sa, y, ya, so, c=None, regularization_factor=1,
                 k_is_positive=False, rank=None):
        # We inherit from the inversion class and create space for
        # the reduced rank solutions.
        Inversion.__init__(self, k, xa, sa, y, ya, so, c,
                           regularization_factor, k_is_positive)

        # We create space for the rank
        self.rank = rank

        # We also want to save the eigendecomposition values
        self.evals_q = None
        self.evals_h = None
        self.evecs = None

        # and for the final solutions
        self.xhat_proj = None
        self.xhat_kproj = None
        self.xhat_fr = None

        self.shat_proj = None
        self.shat_kproj = None

        self.a_proj = None
        self.a_kproj = None

    ########################################
    ### REDUCED RANK INVERSION FUNCTIONS ###
    ########################################
    def get_rank(self, pct_of_info=None, rank=None, snr=None):
        frac = np.cumsum(self.evals_q/self.evals_q.sum())
        if sum(x is not None for x in [pct_of_info, rank, snr]) > 1:
            raise AttributeError('Conflicting arguments provided to determine rank.')
        elif sum(x is not None for x in [pct_of_info, rank, snr]) == 0:
            raise AttributeError('Must provide one of pct_of_info, rank, or snr.')
        elif pct_of_info is not None:
            diff = np.abs(frac - pct_of_info)
            rank = np.argwhere(diff == np.min(diff))[0][0]
            print('Calculated rank from percent of information: %d' % rank)
            print('     Percent of information: %.4f%%' % (100*pct_of_info))
            print('     Signal-to-noise ratio: %.2f' % self.evals_h[rank])
        elif snr is not None:
            diff = np.abs(self.evals_h - snr)
            rank = np.argwhere(diff == np.min(diff))[0][0]
            print('Calculated rank from signal-to-noise ratio : %d' % rank)
            print('     Percent of information: %.4f%%' % (100*frac[rank]))
            print('     Signal-to-noise ratio: %.2f' % snr)
        elif rank is not None:
            print('Using defined rank: %d' % rank)
            print('     Percent of information: %.4f%%' % (100*frac[rank]))
            print('     Signal-to-noise ratio: %.2f' % self.evals_h[rank])
        return rank

    def pph(self):
        # Calculate the prior pre-conditioned Hessian assuming
        # that the errors are diagonal
        if ((self._find_dimension(self.sa) == 1) and
            (self._find_dimension(self.so) == 1)):
            pph = (data.sa**0.5)*data.k.T
            pph = xr.dot(pph, (pph.T/data.so))
        elif self._find_dimension(self.sa) == 1:
            pph = (data.sa**0.5)*data.k.T
            pph = pph @ inv(data.so) @ pph.T
        else:
            pph = data.sa**0.5 @ data.k.T
            pph = pph @ inv(data.so) @ pph.T
        print('Calculated PPH.')
        return pph

    def edecomp(self, pph=None, eval_threshold=None, number_of_evals=None):
        print('... Calculating eigendecomposition ...')
        # Calculate the prior pre-conditioned Hessian
        if pph is None:
            pph = self.pph()

        # Check that the PPH is symmetric
        assert np.allclose(pph, pph.T, rtol=1e-5), \
               'The prior pre-conditioned Hessian is not symmetric.'

        # Perform the eigendecomposition of the prior pre-conditioned Hessian
        # We return the evals of the projection, not of the
        # prior pre-conditioned Hessian.
        if (eval_threshold is None) and (number_of_evals is None):
             evals, evecs = eigh(pph)
        elif (eval_threshold is None):
            n = pph.shape[0]
            evals, evecs = eigh(pph, subset_by_index=[n - number_of_evals,
                                                      n - 1])
        else:
            evals, evecs = eigh(pph, subset_by_value=[eval_threshold, np.inf])
        print('Eigendecomposition complete.')

        # Sort evals and evecs by eval
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:,idx]

        # Force all evals to be non-negative
        if (evals < 0).sum() > 0:
            print('Negative eigenvalues. Maximum negative value is %.2e. Setting negative eigenvalues to zero.' \
                % (evals[evals < 0].min()))
            evals[evals < 0] = 0

        # Check for imaginary eigenvector components and force all
        # eigenvectors to be only the real component.
        if np.any(np.iscomplex(evecs)):
            print('Imaginary eigenvectors exist at index %d of %d. Forcing eigenvectors to real component alone.' \
                  % ((np.where(np.iscomplex(evecs))[1][0] - 1), len(evecs)))
            evecs = np.real(evecs)

        # Saving result to our instance.
        print('Saving eigenvalues and eigenvectors to instance.')
        # self.evals = evals/(1 + evals)
        self.evals_h = evals
        self.evals_q = evals/(1 + evals)
        self.evecs = evecs
        print('... Complete ...\n')

    def projection(self, pct_of_info=None, rank=None, snr=None):
        # Conduct the eigendecomposition of the prior pre-conditioned
        # Hessian
        if ((self.evals_h is None) or
            (self.evals_q is None) or
            (self.evecs is None)):
            self.edecomp()

        # Subset the evecs according to the rank provided.
        rank = self.get_rank(pct_of_info=pct_of_info, rank=rank, snr=snr)
        evecs_subset = self.evecs[:,:rank]

        # Calculate the prolongation and reduction operators and
        # the resulting projection operator.
        prolongation = (evecs_subset.T * self.sa**0.5).T
        reduction = (1/self.sa**0.5) * evecs_subset.T
        projection = prolongation @ reduction

        return rank, prolongation, reduction, projection

    # Need to add in cost function and other information here
    def solve_inversion_proj(self, pct_of_info=None, rank=None):
        print('... Solving projected inversion ...')
        # Conduct the eigendecomposition of the prior pre-conditioned
        # Hessian
        if ((self.evals_h is None) or
            (self.evals_q is None) or
            (self.evecs is None)):
            self.edecomp()

        # Subset the evecs according to the rank provided.
        rank = self.get_rank(pct_of_info=pct_of_info, rank=rank)

        # Calculate a few quantities that will be useful
        sa_sqrt = diags(self.sa**0.5)
        sa_sqrt_inv = diags(1/self.sa**0.5)
        # so = self.regularization_factor*self.so
        so_inv = diags(1/self.so)
        # so_sqrt_inv = diags(1/so**0.5)

        # Subset evecs and evals
        vk = self.evecs[:, :rank]
        # wk = so_sqrt_inv @ self.k @ sa_sqrt @ vk
        lk = self.evals_h[:rank].reshape((1, -1))

        # Make lk into a matrix
        lk = np.repeat(lk, self.nstate, axis=0)

        # Calculate the solutions
        self.xhat_proj = (np.asarray(sa_sqrt
                                    @ ((vk/(1+lk)) @ vk.T)
                                    @ sa_sqrt @ self.k.T @ so_inv
                                    @ self.obs_mod_diff(self.xa))
                         + self.xa)
        self.shat_proj = np.asarray(sa_sqrt
                                    @ (((1/(1+lk))*vk) @ vk.T)
                                    @ sa_sqrt)
        # self.shat_proj = sa_sqrt @ self.shat_proj_sum(rank) @ sa_sqrt
        self.a_proj = np.asarray(sa_sqrt
                                 @ (((lk/(1+lk))*vk) @ vk.T)
                                 @ sa_sqrt_inv)
        print('... Complete ...\n')

    def solve_inversion_kproj(self, pct_of_info=None, rank=None):
        print('... Solving projected Jacobian inversion ...')
        # Get the projected solution
        self.solve_inversion_proj(pct_of_info=pct_of_info, rank=rank)

        # Calculate a few quantities that will be useful
        sa = diags(self.sa)

        # Calculate the solutions
        self.xhat_kproj = self.xhat_proj
        self.shat_kproj = np.asarray((identity(self.nstate) - self.a_proj) @ sa)
        self.a_kproj = self.a_proj
        print('... Complete ...\n')

    def solve_inversion_fr(self, pct_of_info=None, rank=None):
        print('... Solving full rank approximation inversion ...')
        self.solve_inversion_kproj(pct_of_info=pct_of_info, rank=rank)

        so_inv = diags(1/self.so)
        d = self.obs_mod_diff(self.xa)

        self.xhat_fr = self.shat_kproj @ self.k.T @ so_inv @ d
        print('... Complete ...\n')

    ##########################
    ### PLOTTING FUNCTIONS ###
    ##########################

    def plot_info_frac(self, aspect=1.75, relative=True, **kw):
        fig_kwargs = kw.pop('fig_kwargs', {})
        fig, ax = fp.get_figax(aspect=aspect, **fig_kwargs)

        label = kw.pop('label', '')
        color = kw.pop('color', plt.cm.get_cmap('inferno')(5))
        ls = kw.pop('ls', '-')
        lw = kw.pop('lw', 3)
        text = kw.pop('text', True)
        if kw:
            raise TypeError('Unexpected kwargs provided: %s' % list(kw.keys()))

        frac = np.cumsum(self.evals_q/self.evals_q.sum())
        snr_idx = np.argwhere(self.evals_q >= 0.5)[-1][0]
        ax.plot(frac, label=label, c=color, ls=ls, lw=lw)

        if text:
            ax.scatter(snr_idx, frac[snr_idx], s=10*config.SCALE, c=color)
            ax.text(snr_idx + self.nstate*0.05, frac[snr_idx],
                    r'SNR $\approx$ 1',
                    ha='left', va='top',
                    fontsize=config.LABEL_FONTSIZE*config.SCALE,
                    color=color)
            ax.text(snr_idx + self.nstate*0.05, frac[snr_idx] - 0.075,
                    'n = %d' % snr_idx,
                    ha='left', va='top',
                    fontsize=config.LABEL_FONTSIZE*config.SCALE,
                    color=color)
            ax.text(snr_idx + self.nstate*0.05, frac[snr_idx] - 0.15,
                    r'$f_{DOFS}$ = %.2f' % frac[snr_idx],
                    ha='left', va='top',
                    fontsize=config.LABEL_FONTSIZE*config.SCALE,
                    color=color)

        ax = fp.add_legend(ax)
        ax = fp.add_labels(ax,
                           xlabel='Eigenvector Index',
                           ylabel='Fraction of DOFS')
        ax = fp.add_title(ax, title='Information Content Spectrum')

        return fig, ax

class ReducedRankJacobian(ReducedRankInversion):
    def __init__(self, k, xa, sa, y, ya, so, c=None, regularization_factor=1,
                 k_is_positive=False, rank=None):
        # Inherit from the parent class.
        ReducedRankInversion.__init__(self, k, xa, sa, y, ya, so, c,
                                      regularization_factor, k_is_positive,
                                      rank)
        self.model_runs = 0

    # Rewrite to take any prolongation matrix?
    def update_jacobian(self, forward_model,
                        pct_of_info=None, rank=None, snr=None,
                        prolongation=None, reduction=None):
        # Retrieve the prolongation operator associated with this
        # instance of the Jacobian for the rank specified by the
        # function call. These are the eigenvectors that we will perturb.
        if sum(x is not None
               for x in [pct_of_info, rank, snr, prolongation]) > 1:
            raise AttributeError('Conflicting arguments provided to determine rank.')
        elif sum(x is not None
                 for x in [pct_of_info, rank, snr, prolongation]) == 0:
            raise AttributeError('Must provide one of pct_of_info, rank, or prolongation.')
        elif (((prolongation is not None) and (reduction is None)) or
              ((prolongation is None) and (reduction is not None))):
            raise AttributeError('Only one of prolongation or reduction is provided.')
        elif (prolongation is not None) and (reduction is not None):
            print('Using given prolongation and reduction.')
        else:
            print('Calculating prolongation and reduction.')
            _, prolongation, reduction, _ = self.projection(pct_of_info=pct_of_info,
                                                            rank=rank,
                                                            snr=snr)

        # Run the perturbation runs
        perturbations = forward_model @ prolongation

        # Project to the full dimension
        k = perturbations @ reduction

        # Save to a new instance
        new = ReducedRankJacobian(k=k,
                                  xa=copy.deepcopy(self.xa),
                                  sa=copy.deepcopy(self.sa),
                                  y=copy.deepcopy(self.y),
                                  ya=copy.deepcopy(self.ya),
                                  so=copy.deepcopy(self.so),
                                  regularization_factor=self.regularization_factor)
        new.model_runs = copy.deepcopy(self.model_runs) + prolongation.shape[1]

        # Do the eigendecomposition
        new.edecomp()

        # And solve the inversion
        new.solve_inversion()
        # new.solve_inversion_kproj(rank=floor(rank/2))

        return new

    def filter(self, true, mask):
        self_f = copy.deepcopy(self)
        true_f = copy.deepcopy(true)

        skeys = [k for k in dir(self_f) if k[:4] == 'shat']
        xkeys = [k for k in dir(self_f) if k[:4] == 'xhat']
        akeys = [k for k in dir(self_f) if (k == 'a') or (k[:2] == 'a_')]

        for obj in [true_f, self_f]:
            # update jacobian and DOFS
            setattr(obj, 'k', getattr(obj, 'k')[:, mask])
            setattr(obj, 'dofs', getattr(obj, 'dofs')[mask])

            for keylist in [skeys, xkeys, akeys]:
                for k in keylist:
                    try:
                        if getattr(obj, k).ndim == 1:
                            # update true_f and self_f posterior mean
                            setattr(obj, k, getattr(obj, k)[mask])
                        else:
                            # update true_f and self_f posterior variance
                            setattr(obj, k, getattr(obj, k)[mask, :][:, mask])
                    except AttributeError:
                        pass

        # some plotting functions
        true_f.xhat_long = copy.deepcopy(true.xhat)
        true_f.xhat_long[~mask] = 1
        self_f.xhat_long = copy.deepcopy(self.xhat)
        self_f.xhat_long[~mask] = 1

        return self_f, true_f

class ReducedDimensionJacobian(ReducedRankInversion):
    def __init__(self, k, xa, sa, y, ya, so, c=None, regularization_factor=1,
                 reduced_memory=False, available_memory_GB=None,
                 k_is_positive=False, rank=None):

        # state_vector    A vector containing an index for each state vector
        #                 element (necessary for multiscale grid methods).

        # Inherit from the parent class.
        ReducedRankInversion.__init__(self, k, xa, sa, y, ya, so, c,
                                      regularization_factor, reduced_memory,
                                      available_memory_GB, k_is_positive,
                                      rank)

        self.state_vector = np.arange(1, self.nstate+1, 1)
        self.perturbed_cells = np.array([])
        self.model_runs = 0

    def merge_cells_kmeans(self, label_idx, clusters_plot, cluster_size):
        labels = np.zeros(self.nstate)
        labels[label_idx] = label_idx + 1
        # the +1 here is countered by a -1 later--this is a fix for
        # pythonic indexing
        labels = ip.match_data_to_clusters(labels,
                                               clusters_plot)
        labels = labels.to_dataframe('labels').reset_index()
        labels = labels[labels['labels'] > 0]
        labels['labels'] -= 1

        # Now do kmeans clustering
        n_clusters = int(len(label_idx)/cluster_size)
        labels_new = KMeans(n_clusters=n_clusters,
                            random_state=0)
        labels_new = labels_new.fit(labels[['lat', 'lon']])

        # Print out some information
        label_stats = np.unique(labels_new.labels_, return_counts=True)
        print('Number of clusters: %d' % len(label_stats[0]))
        print('Cluster size: %d' % cluster_size)
        print('Maximum number of grid boxes in a cluster: %d' \
              % max(label_stats[1]))
        print('Average number of grid boxes in a cluster: %.2f' \
              % np.mean(label_stats[1]))
        print('...')

        # save the information
        labels = labels.assign(new_labels=labels_new.labels_+1)
        labels[['labels', 'new_labels']] = labels[['labels',
                                                   'new_labels']].astype(int)

        return labels

    def aggregate_cells_kmeans(self, clusters_plot,
                               n_cells, n_cluster_size=None,
                               dofs_e=None):
        print('... Generating multiscale grid ...')

        # If first aggregation
        if len(self.state_vector) == self.nstate:
            # Get the indices associated with the most significant
            # grid boxes
            sig_idx = self.dofs.argsort()[::-1]

            new_sv = np.zeros(self.nstate)

            # Iterate through n_cells
            n_cells = np.append(0, n_cells)
            nidx = np.cumsum(n_cells)
            for i, n in enumerate(n_cells[1:]):
                # get cluster size
                if n_cluster_size is None:
                    cluster_size = i+2
                else:
                    cluster_size = n_cluster_size[i]

                # get the indices of interest
                sub_sig_idx = sig_idx[nidx[i]:nidx[i+1]]

                new_labels = self.merge_cells_kmeans(sub_sig_idx,
                                                     clusters_plot,
                                                     cluster_size)

                new_sv[new_labels['labels']] = new_labels['new_labels']+new_sv.max()
                added_model_runs = len(np.unique(new_sv))

        # If second aggregation (i.e. disaggregation)
        else:
            # expected DOFS
            # Group the previous estimate of the DOFS by the state vector
            dofs_e = pd.DataFrame(copy.deepcopy(dofs_e))
            dofs_e = dofs_e.groupby(copy.deepcopy(self.state_vector)).sum()
            dofs_e = np.array(dofs_e).reshape(-1,)

            # Normallize by the number of grid cellls in each cluster
            _, count = np.unique(self.state_vector, return_counts=True)
            dofs_per_cell_e = dofs_e/count

            # Calculate the difference (absolute and relative)
            dofs_diff = self.dofs - dofs_per_cell_e
            dofs_diff_rel = dofs_diff/dofs_per_cell_e

            # Ignore grid cells that are already at native resolution
            dofs_diff[count == 1] = 0

            # Get the indices associated with the largest differences
            sig_idx = dofs_diff.argsort()[::-1] + 1

            new_sv = copy.deepcopy(self.state_vector)
            count = 0
            i = 0
            while count <= n_cells[0]:
                idx = np.where(sig_idx[i] == self.state_vector)[0]
                renumber_idx = np.where(sig_idx[i] < self.state_vector)[0]
                new_sv[renumber_idx] += len(idx) - 1
                new_sv[idx] = np.arange(new_sv[idx[0]],
                                        new_sv[idx[0]] + len(idx))
                count += len(idx) - 1
                i += 1
            print('%d cells disaggregated' % (i-1))
            added_model_runs = count
            # now renumber

        # if we're at the point of disaggregating cells
        # take the largest sig_idx[1]
        # set all indices equal to new, unique values
        # count the number of new indices
        # while less than n_cells, continue
        print('Number of state vector elements: %d' \
              % len(np.unique(new_sv)))
        print('... Complete ...\n')

        return new_sv, added_model_runs


    def disaggregate_k_ms(self):
        # Create space for the full resolution Jacobian
        k_fr = np.zeros((self.nobs, len(self.state_vector)))
        for i in range(1, self.nstate+1):
            # get the column indices for the new Jacobian
            cols_idx = np.where(self.state_vector == i)[0]

            # get the Jacobian values
            ki = self.k[:, i-1]/len(cols_idx)
            ki = np.tile(ki.reshape(-1, 1), (1, len(cols_idx)))

            # fill in
            k_fr[:, cols_idx] = ki
        return k_fr

    def calculate_k(self, forward_model):
        k_ms = pd.DataFrame(forward_model)
        k_ms = k_ms.groupby(self.state_vector, axis=1).sum()
        self.k = np.array(k_ms)

    def calculate_prior(self, xa_abs, sa):
        xa_abs_ms = pd.DataFrame(xa_abs).groupby(self.state_vector).sum()

        sa_abs = pd.DataFrame(sa*xa_abs)
        sa_abs_ms = (sa_abs**2).groupby(self.state_vector).sum()**0.5
        sa_ms = sa_abs_ms/xa_abs_ms

        self.xa_abs = np.array(xa_abs_ms).reshape(-1,)
        self.xa = np.ones(self.nstate).reshape(-1,)
        self.sa = np.array(sa_ms).reshape(-1,)

    def update_jacobian_rd(self, forward_model, xa_abs, sa, clusters_plot,
                           pct_of_info=None, rank=None, snr=None,
                           n_cells=[100, 200],
                           n_cluster_size=[1, 2],
                           dofs_e=None,
                           k_base=None):#, #threshold=0,
                        # perturbation_factor=0.5):
        '''
        This function generates a multi-scale Jacobian on the basis
        of the information content, as given by the diagonal elements
        of the averaging kernel. It maintains the native resolution
        for the grid cells with the highest information content while
        aggregating together grid cells with lower information content
        using K means clustering. The user must provide the number of
        desired number of grid cells and the number of grid cells per
        cluster for each aggregation level. It accepts the following
        arguments:
            forward_model       the true Jacobian
            clusters_plot       the mapping from the grid cells to state
                                vector number
            pct_of_info         currently not used
            rank                currently not used
            snr                 currently not used
            n_cells             the number of native resolution grid
                                cells to be used in the aggregation
                                scheme (integer or list of integers);
                                defaults to [100, 200]
            n_cluster_size      the number of native resolution grid
                                cells to aggregate together at each level
                                of aggregation; defaults to [1, 2]
            k_base              currently defaults to previous k
        Example:
            Passing n_cells=[100, 200], n_cluster_size=[1, 2] will
            generate a state vector where the 100 grid cells with highest
            information content maintain the native resolution (i.e.
            cluster size = 1) and the 200 grid cells with next highest
            information content will be aggregated into clusters with size
            2. The final state vector will have dimension 200.
        '''

        if k_base is None:
            k_base = copy.deepcopy(self.k)

        # We start by creating a new instance of the class. This
        # is where
        new = ReducedRankJacobian(k=k_base,
                                  xa=copy.deepcopy(self.xa),
                                  sa=copy.deepcopy(self.sa),
                                  y=copy.deepcopy(self.y),
                                  ya=copy.deepcopy(self.ya),
                                  so=copy.deepcopy(self.so))
        new.state_vector = copy.deepcopy(self.state_vector)
        new.dofs = copy.deepcopy(self.dofs)
        new.model_runs = copy.deepcopy(self.model_runs)
        new.regularization_factor = copy.deepcopy(self.regularization_factor)
        _, counts = np.unique(self.state_vector, return_counts=True)

        # If previously optimized, set significance to 0
        # if len(self.perturbed_cells) > 0:
        if np.any(counts > 1):
            print('Ignoring previously optimized grid cells.')
            # significance[self.perturbed_cells] = 0
            new.dofs[counts == 1] = 0

        # We need the new state vector first. This gives us the
        # clusterings of the base resolution state vector
        # elements as dictated by n_cells and n_cluster_size.
        new.state_vector, new_runs = new.aggregate_cells_kmeans(clusters_plot,
                                                       n_cells=n_cells,
                                                       n_cluster_size=  n_cluster_size,
                                                       dofs_e=dofs_e)
        new.model_runs += new_runs

        # We calculate the number of model runs and the counts of
        # each cluster
        elements, counts = np.unique(new.state_vector, return_counts=True)

        # # Now update the Jacobian
        new.nstate = len(elements)
        new.calculate_k(forward_model)
        new.calculate_prior(xa_abs=xa_abs, sa=sa)

        # Adjust the regularization_factor
        new.regularization_factor = self.regularization_factor*new.nstate/self.nstate
        # THIS NEEDS TO BE FIXED NOW THAT WE DEFINE SO/REG FACTOR IN
        # INITIALIZATION

        # Update the value of c in the new instance
        new.calculate_c()

        # And do the eigendecomposition
        new.edecomp()

        # And solve the inversion
        new.solve_inversion()

        # And save a long xhat/shat/avker (start by saving them as
        # an unphysical value so that we can check for errors)
        new.xhat_long = np.ones(len(new.state_vector))
        new.dofs_long = np.zeros(len(new.state_vector))
        for i in range(1, new.nstate + 1):
            idx = np.where(new.state_vector == i)[0]
            new.xhat_long[idx] = new.xhat[i - 1]
            new.dofs_long[idx] = new.dofs[i - 1]

        print('NUMBER OF MODEL RUNS : %d' % new.model_runs)

        return new

    def plot_multiscale_grid(self, clusters_plot, **kw):
        # Get KW
        title = kw.pop('title', '')
        fig_kwargs = kw.pop('fig_kwargs', {})
        title_kwargs = kw.pop('title_kwargs', {})
        map_kwargs = kw.pop('map_kwargs', {})
        kw['colors'] = kw.pop('colors', 'black')
        kw['linewidths'] = kw.pop('linewidths', 1)

        # Plot
        nstate = len(np.unique(self.state_vector)[1:])
        data = ip.match_data_to_clusters(self.state_vector,
                                           clusters_plot, default_value=0)
        data_zoomed = zoom(data.values, 50, order=0, mode='nearest')
        fig, ax = fp.get_figax(maps=True, lats=data.lat, lons=data.lon,
                               **fig_kwargs)
        ax.contour(data_zoomed, levels=np.arange(0, nstate, 1),
                   extent=[data.lon.min(), data.lon.max(),
                           data.lat.min(), data.lat.max()],
                   **kw)
        ax = fp.add_title(ax, title, **title_kwargs)
        ax = fp.format_map(ax, data.lat, data.lon, **map_kwargs)

        return fig, ax

class ReducedMemoryInversion(ReducedRankJacobian):
    def __init__(self, k, xa, sa, y, ya, so, c=None, regularization_factor=1,
                 reduced_memory=True, available_memory_GB=None,
                 k_is_positive=False, rank=None):

        '''
            reduced_memory           This flag (default false) will determine
                                     whether or not to read in the file or
                                     arrays provided in chunks.
            available_memory_GB      If reduced_memory is True, both the number
                                     of cores and the memory available (in
                                     gigabytes) must be provided.
            k_is_positive            A flag indicating whether the Jacobian
                                     has already been checked for negative
                                     elements. Default is False.
        '''

        # Check that available memory is provided
        assert available_memory_GB is not None, \
               'Available memory is not provided.'

        # Print out a reminder to set the number of threads
        print('*'*60)
        print('NOTE: REDUCED_MEMORY = TRUE')
        print('It is recommended that OMP_NUM_THREADS be set to half the')
        print('available cores [N/2] by running')
        print('  >> export OMP_NUM_THREADS=[N/2]')
        print('After running this command, rerun any scripts that use the')
        print('Inversion class.')
        print('*'*60)
        print('')

        # Set initial chunk values
        chunks = {'nstate' : -1, 'nobs' : None}

        # Before inheriting from the parent class, we are going to load
        # all of the items using chunks.

        # Read in the prior elements first because that allows us to calculate
        # nstate and thus the chunk size in that dimension.
        
        # Get dimensions for the input (allowing 1D vector inputs for sa and so)
        dims = self._get_dims(sa,so)
        
        print('Loading the prior and prior error.')
        xa = self.read(xa, dims=Inversion.dims['xa'], chunks=chunks)
        sa = self.read(sa, dims=Inversion.dims['sa'], chunks=chunks)

        # Save out the state vector dimension
        self.nstate = xa.shape[0]
        print(f'State vector dimension : {self.nstate}\n')

        # Update the chunks for the nobs dimension accordingly
        max_chunk_size = gc.calculate_chunk_size(available_memory_GB)
        chunks['nobs'] = int(max_chunk_size/self.nstate)

        # Load observational data
        k = self.read(k, dims=Inversion.dims['k'], chunks=chunks)
        y = self.read(y, dims=Inversion.dims['y'], chunks=chunks)
        ya = self.read(ya, dims=Inversion.dims['ya'], chunks=chunks)
        so = self.read(so, dims=Inversion.dims['so'], chunks=chunks)

        # Inherit from the parent class.
        ReducedRankJacobian.__init__(self, k, xa, sa, y, ya, so, c,
                                     regularization_factor, reduced_memory,
                                     available_memory_GB, k_is_positive,
                                     rank)

        # Save out chunks
        self.chunks = chunks


    #########################
    ### UTILITY FUNCTIONS ###
    #########################
    def read(self, item, dims=None, chunks={}, **kwargs):
        # If item is a string or a list, load the file
        if type(item) in [str, list]:
            item = self._load(item, dims=dims, chunks=chunks,
                              **kwargs)

        # Force the items to be dataarrays
        if type(item) != xr.core.dataarray.DataArray:
            item = self._to_dataarray(item, dims=dims, chunks=chunks)

        return item

    @staticmethod
    def _load(item, dims, chunks, **kwargs):
        # This function is only called by read()
        # Add chunks to kwargs.
        kwargs['chunks'] = {k : chunks[k] for k in dims}
        if type(item) == list:
            # If it's a list, set some specific settings for open_mfdataset
            # This currently assumes that the coords are not properly set,
            # which was an oopsy when making the K0 datasets.
            kwargs['combine'] = 'nested'
            kwargs['concat_dim'] = dims
        else:
            item = [item]

        item = gc.read_file(*item, **kwargs)
        return item

    @staticmethod
    def _to_dataarray(item, dims, chunks):
        # If it's a dataset, require that there be only one variable
        if type(item) == xr.core.dataset.Dataset:
            variables = list(item.keys())
            assert len(variables) == 1, \
                  'A dataset with multiple variables was provided. This \
                   class requires that any datasets contain only one \
                   variable.'
            item = item[variables[0]]
        # Else convert to a dataarray
        else:
            assert dims is not None, \
                   'Creating an xarray dataset and dims is not provided.'
            chunks = {k : chunks[k] for k in dims}
            item = xr.DataArray(item, dims=tuple(dims))
            if len(chunks) > 0:
                item = item.chunk(chunks)
        return item

    def save(self, attribute:str, file_name=None):
        item = getattr(self, attribute)

        # Get file name
        if file_name is None:
            file_name = attribute

        with ProgressBar():
                item.to_netcdf(f'{file_name}.nc')

    ###########################
    ### INVERSION FUNCTIONS ###
    ###########################
    def solve_inversion(self):
        ...

import xarray as xr
import numpy as np
# from numpy import diag as diags
# from numpy import identity
from numpy.linalg import inv, norm #, eigh
from scipy.sparse import diags, identity
from scipy.stats import linregress
from scipy.linalg import eigh
import pandas as pd
import copy

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
    def __init__(self, k, xa, sa_vec, y, y_base, so_vec,
                 rf=1, latres=1, lonres=1.25):
        '''
        Define an inversion object with the following required
        inputs:
            k               The Jacobian matrix for the forward model,
                            K = dy/dx, which represents the sensitivity of the
                            simulated observations to each element of the
                            state vector
            xa              The prior for the state vector
            sa_vec          The prior error variances as a vector. This class
                            is not currently written to accept error
                            covariances (such a change would be simple to
                            implement).
            y               The observations
            y_base          The simulated observations generated from the prior
                            state vector
            so_vec          The observational error variances as a vector,
                            including errors from the forward model and the
                            observations. This class is not currently written
                            to accept error covariances, though such a change
                            would be simple to implement

        The object also accepts the following inputs:
            rf              A regularization factor gamma defined in the cost
                            function as
                                J(x) = (x-xa)T Sa (x-xa) + rf(y-Kx)T So (y-Kx).
                            The regularization factor changes the weighting of
                            the observational term relative to the prior term.
                            The rf is functionally equivalent to dividing the
                            observational error variances by rf.
            latres          The latitudinal resolution of the state vector
            lonres          The longitudinal resolution of the state vector

        The object then defines the following:
            nstate          The dimension of the state vector
            nobs            The dimension of the observation vector
            state_vector    A vector containing an index for each state vector
                            element (necessary for multiscale grid methods).
            c               The constant defined as y = Kxa + c
            xhat            Empty, held for the posterior state vector
            shat            Empty, held for the posterior error
            a               Empty, held for the averaging kernel, a measure
                            of the information content of the inverse system
            y_out           Empty, the updated simulated observations, defined
                            as y = Kxhat + c
        '''
        print('... Initializing inversion object ...')

        # Check that the data are all the same types
        assert all(isinstance(z, np.ndarray)
                   for z in [k, xa, sa_vec, y, so_vec]), \
               'Input types aren\'t all numpy arrays.'

        # Define the state and observational dimensions
        self.nstate = xa.shape[0]
        self.nobs = y.shape[0]
        self.latres = latres
        self.lonres = lonres
        self.state_vector = np.arange(1, self.nstate+1, 1)

        # Check whether all inputs have the right dimensions
        assert k.shape[1] == self.nstate, \
               'Dimension mismatch: Jacobian and prior.'
        assert k.shape[0] == self.nobs, \
               'Dimension mismatch: Jacobian and observations.'
        assert so_vec.shape[0] == self.nobs, \
               'Dimension mismatch: observational error'
        assert sa_vec.shape[0] == self.nstate, \
               'Dimension mismatch: prior error.'

        # If everything works out, then we create the instance.
        self.k = k
        self.xa = xa
        self.sa_vec = sa_vec
        self.y = y
        self.y_base = y_base
        self.so_vec = so_vec
        self.rf = rf

        # Force k to be positive
        if np.any(self.k < 0):
            print('Forcing negative values of the Jacobian to 0.')
            self.k[self.k < 0] = 0

        # Solve for the constant c.
        self.c = self.calculate_c()

        # Now create some holding spaces for values that may be filled
        # in the course of solving the inversion.
        self.xhat = None
        self.shat = None
        self.a = None
        self.y_out = None

        print('... Complete ...\n')

    ####################################
    ### STANDARD INVERSION FUNCTIONS ###
    ####################################

    def calculate_c(self):
        '''
        Calculate c for the forward model, defined as ybase = Kxa + c.
        Save c as an element of the object.
        '''
        c = self.y_base - self.k @ self.xa
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
            J(x) = (x - xa)T Sa (x-xa) + rf(y - Kx)T So (y - Kx)
        for a given x. Prints out that value and the contributions from
        the emission and observational terms.

        Parameters:
            x      The state vector at which to evaluate the cost function
        Returns:
            cost   The value of the cost function at x
        '''

        # Calculate the observational component of the cost function
        cost_obs = self.obs_mod_diff(x).T \
                   @ diags(self.rf/self.so_vec) @ self.obs_mod_diff(x)

        # Calculate the emissions/prior component of the cost function
        cost_emi = (x - self.xa).T @ diags(1/self.sa_vec) @ (x - self.xa)

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

        # We use the inverse of both the prior and observational
        # error covariance matrices, so we save those as separate variables.
        # Here we convert the variance vectors into diagonal covariance
        # matrices. We also apply the regularization factor rf to the
        # observational error covariance.
        # Note: This would change if error variances were redefined as
        # covariance matrices
        so_inv = diags(self.rf/self.so_vec)
        sa_inv = diags(1/self.sa_vec)

        # Calculate the cost function at the prior.
        print('Calculating the cost function at the prior mean.')
        cost_prior = self.cost_func(self.xa)

        # Calculate the posterior error.
        print('Calculating the posterior error.')
        self.shat = np.asarray(inv(self.k.T @ so_inv @ self.k + sa_inv))

        # Calculate the posterior mean
        print('Calculating the posterior mean.')
        gain = np.asarray(self.shat @ self.k.T @ so_inv)
        self.xhat = self.xa + (gain @ self.obs_mod_diff(self.xa))

        # Calculate the cost function at the posterior. Also
        # calculate the number of negative cells as an indicator of
        # inversion success.
        print('Calculating the cost function at the posterior mean.')
        cost_post = self.cost_func(self.xhat)
        print('     Negative cells: %d' % self.xhat[self.xhat < 0].sum())

        # Calculate the averaging kernel.
        print('Calculating the averaging kernel.')
        self.a = np.asarray(identity(self.nstate) \
                            - self.shat @ sa_inv)
        self.dofs = np.diag(self.a)
        print('     DOFS: %.2f' % np.trace(self.a))

        # Calculate the new set of modeled observations.
        print('Calculating updated modeled observations.')
        self.y_out = self.k @ self.xhat + self.c

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
        data = ip.match_data_to_clusters(data, clusters_plot, default_value)

        # Plot
        fig, ax, c = ip.plot_state_format(data, default_value, cbar, **kw)
        return fig, ax, c

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

class ReducedMemoryInversion(Inversion):
    def __init__(self, k_files, xa, sa_vec, y, y_base, so_vec,
                 rf=1, latres=1, lonres=1.25):
        print('... Initializing reduced memory inversion object ...')

        # Check that the data are all the same types
        assert all(isinstance(z, np.ndarray)
                   for z in [xa, sa_vec, y, so_vec]), \
               'Input types aren\'t all numpy arrays.'

        assert isinstance(k, list), 'Jacobian is not a list of files.'

        # Define the state and observational dimensions
        self.nstate = xa.shape[0]
        self.nobs = y.shape[0]
        self.latres = latres
        self.lonres = lonres
        self.state_vector = np.arange(1, self.nstate+1, 1)

        # Check whether all inputs have the right dimensions
        assert k.shape[1] == self.nstate, \
               'Dimension mismatch: Jacobian and prior.'
        assert k.shape[0] == self.nobs, \
               'Dimension mismatch: Jacobian and observations.'
        assert so_vec.shape[0] == self.nobs, \
               'Dimension mismatch: observational error'
        assert sa_vec.shape[0] == self.nstate, \
               'Dimension mismatch: prior error.'

        # If everything works out, then we create the instance.
        self.k = k
        self.xa = xa
        self.sa_vec = sa_vec
        self.y = y
        self.y_base = y_base
        self.so_vec = so_vec
        self.rf = rf

        # # Solve for the constant c.
        # self.calculate_c()

        # Now create some holding spaces for values that may be filled
        # in the course of solving the inversion.
        self.xhat = None
        self.shat = None
        self.a = None
        self.y_out = None

        print('... Complete ...\n')

    ####################################
    ### STANDARD INVERSION FUNCTIONS ###
    ####################################
    @staticmethod
    def open_k(file_name):
        try:
            k = gc.load_obj(file_name)
            return k
        except FileNotFoundError:
            print(f'{file_name} not found.')

    def calculate_c(self):
        '''
        Calculate c for the forward model, defined as ybase = Kxa + c.
        Save c as an element of the object.
        '''
        c =
        for file_name in self.k:
            k = self.open_k(file_name)
            c = self.y_base - self.k @ self.xa

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
            J(x) = (x - xa)T Sa (x-xa) + rf(y - Kx)T So (y - Kx)
        for a given x. Prints out that value and the contributions from
        the emission and observational terms.

        Parameters:
            x      The state vector at which to evaluate the cost function
        Returns:
            cost   The value of the cost function at x
        '''

        # Calculate the observational component of the cost function
        cost_obs = self.obs_mod_diff(x).T \
                   @ diags(self.rf/self.so_vec) @ self.obs_mod_diff(x)

        # Calculate the emissions/prior component of the cost function
        cost_emi = (x - self.xa).T @ diags(1/self.sa_vec) @ (x - self.xa)

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

        # We use the inverse of both the prior and observational
        # error covariance matrices, so we save those as separate variables.
        # Here we convert the variance vectors into diagonal covariance
        # matrices. We also apply the regularization factor rf to the
        # observational error covariance.
        # Note: This would change if error variances were redefined as
        # covariance matrices
        so_inv = diags(self.rf/self.so_vec)
        sa_inv = diags(1/self.sa_vec)

        # Calculate the cost function at the prior.
        print('Calculating the cost function at the prior mean.')
        cost_prior = self.cost_func(self.xa)

        # Calculate the posterior error.
        print('Calculating the posterior error.')
        self.shat = np.asarray(inv(self.k.T @ so_inv @ self.k + sa_inv))

        # Calculate the posterior mean
        print('Calculating the posterior mean.')
        gain = np.asarray(self.shat @ self.k.T @ so_inv)
        self.xhat = self.xa + (gain @ self.obs_mod_diff(self.xa))

        # Calculate the cost function at the posterior. Also
        # calculate the number of negative cells as an indicator of
        # inversion success.
        print('Calculating the cost function at the posterior mean.')
        cost_post = self.cost_func(self.xhat)
        print('     Negative cells: %d' % self.xhat[self.xhat < 0].sum())

        # Calculate the averaging kernel.
        print('Calculating the averaging kernel.')
        self.a = np.asarray(identity(self.nstate) \
                            - self.shat @ sa_inv)
        self.dofs = np.diag(self.a)
        print('     DOFS: %.2f' % np.trace(self.a))

        # Calculate the new set of modeled observations.
        print('Calculating updated modeled observations.')
        self.y_out = self.k @ self.xhat + self.c

        print('... Complete ...\n')


class ReducedRankInversion(Inversion, ReducedMemoryInversion):
    # class variables shared by all instances

    def __init__(self, k, xa, sa_vec, y, y_base, so_vec):
        # We inherit from the inversion class and create space for
        # the reduced rank solutions.
        Inversion.__init__(self, k, xa, sa_vec, y, y_base, so_vec)

        # We create space for the rank
        self.rank = None

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
        sa_sqrt = self.sa_vec**0.5
        so_inv = self.rf/self.so_vec
        pph = (self.sa_vec**0.5)*self.k.T
        pph = np.dot(pph, ((self.rf/self.so_vec)*pph.T))
        print('Calculated PPH.')
        return pph

    def edecomp(self, eval_threshold=None, number_of_evals=None):
        print('... Calculating eigendecomposition ...')
        # Calculate pph and require that it be symmetric
        pph = self.pph()
        assert np.allclose(pph, pph.T, rtol=1e-5), \
               'The prior pre-conditioned Hessian is not symmetric.'

        # Perform the eigendecomposition of the prior
        # pre-conditioned Hessian
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
        prolongation = (evecs_subset.T * self.sa_vec**0.5).T
        reduction = (1/self.sa_vec**0.5) * evecs_subset.T
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
        sa_sqrt = diags(self.sa_vec**0.5)
        sa_sqrt_inv = diags(1/self.sa_vec**0.5)
        # so_vec = self.rf*self.so_vec
        so_inv = diags(self.rf/self.so_vec)
        # so_sqrt_inv = diags(1/so_vec**0.5)

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
        sa = diags(self.sa_vec)

        # Calculate the solutions
        self.xhat_kproj = self.xhat_proj
        self.shat_kproj = np.asarray((identity(self.nstate) - self.a_proj) @ sa)
        self.a_kproj = self.a_proj
        print('... Complete ...\n')

    def solve_inversion_fr(self, pct_of_info=None, rank=None):
        print('... Solving full rank approximation inversion ...')
        self.solve_inversion_kproj(pct_of_info=pct_of_info, rank=rank)

        so_inv = diags(self.rf/self.so_vec)
        d = self.obs_mod_diff(self.xa)

        self.xhat_fr = self.shat_kproj @ self.k.T @ so_inv @ d
        print('... Complete ...\n')


    ##########################
    ########## ERROR #########
    ##########################

    def calc_error(self, attribute, compare_data):
        '''
        self = truth  (x axis)
        compare_data = y axis
        '''
        e = compare_data
        t = getattr(self, attribute)
        err_abs = np.linalg.norm(t - e)
        err_rel = err_abs/np.linalg.norm(t)
        return err_abs, err_rel

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

    @staticmethod
    def calc_stats(xdata, ydata):
        m, b, r, p, err = linregress(xdata.flatten(),
                                     ydata.flatten())
        return m, b, r

    def plot_comparison_dict(self, xdata, compare_data, **kw):
        fig_kwargs = kw.pop('fig_kwargs', {})
        fig, ax = fp.get_figax(**fig_kwargs)
        ax.set_aspect('equal')

        # We need to know how many data sets were passed
        n = len(compare_data)
        cmap = kw.pop('cmap', 'inferno')

        # Plot data
        count = 0
        for k, ydata in compare_data.items():
            ax.scatter(xdata, ydata,
                       alpha=0.5, s=5*fp.SCALE,
                       c=color(count, cmap=cmap, lut=n))
            count += 1

        # Color bar (always True)
        cax = fp.add_cax(fig, ax)
        cbar_ticklabels = kw.pop('cbar_ticklabels',
                                 list(compare_data.keys()))
        norm = colors.Normalize(vmin=0, vmax=n)
        cbar = colorbar.ColorbarBase(cax,
                                     cmap=plt.cm.get_cmap(cmap, lut=n),
                                     norm=norm)
        cbar.set_ticks(0.5 + np.arange(0,n+1))
        cbar.set_ticklabels(cbar_ticklabels)
        cbar = format_cbar(cbar)

        return fig, ax, cbar

    def plot_comparison_hexbin(self, xdata, compare_data,
                               cbar, stats, **kw):
        cbar_kwargs = kw.pop('cbar_kwargs', {})
        fig_kwargs = kw.pop('fig_kwargs', {})
        fig, ax = fp.get_figax(**fig_kwargs)
        ax.set_aspect('equal')

        # Get data limits
        xlim, ylim, xy, dmin, dmax = fp.get_square_limits(xdata, compare_data)

        # Set bins and gridsize for hexbin
        bin_max = len(self.xhat)/10
        round_by = len(str(len(self.xhat)/20).split('.')[0]) - 1
        bins = np.arange(0, int(round(bin_max, -round_by)))
        gridsize = math.floor((dmax - dmin)/(xy[1] - xy[0])*40)

        # Plot hexbin
        c = ax.hexbin(xdata, compare_data,
                      cmap=fp.cmap_trans('plasma_r'),
                      bins=bins,
                      gridsize=gridsize)

        # Aesthetics
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Print information about R2 on the plot
        if stats:
            _, _, r = self.calc_stats(xdata, compare_data)
            if r**2 <= 0.99:
                ax.text(0.05, 0.85,
                        r'R = %.2f' % r,
                        fontsize=config.LABEL_FONTSIZE*config.SCALE,
                        transform=ax.transAxes)
            else:
                ax.text(0.05, 0.85,
                        r'R $>$ 0.99',
                        fontsize=config.LABEL_FONTSIZE*fp.SCALE,
                        transform=ax.transAxes)

        if cbar:
            cbar_title = cbar_kwargs.pop('title', '')
            cax = fp.add_cax(fig, ax)
            cbar = fig.colorbar(c, cax=cax, boundaries=bins,
                                **cbar_kwargs)
            cbar = fp.format_cbar(cbar, cbar_title)
            return fig, ax, cbar
        else:
            return fig, ax, c

    def plot_comparison(self, attribute, compare_data,
                        cbar=True,
                        stats=True, **kw):
        # Get x data
        xdata = getattr(self, attribute)

        # Get other plot labels
        xlabel = kw.pop('xlabel', 'Native Resolution')
        ylabel = kw.pop('ylabel', 'Estimate')
        label_kwargs = kw.pop('label_kwargs', {})
        title = kw.pop('title', 'Estimated vs. Native Resolution ' + attribute)
        title_kwargs = kw.pop('title_kwargs', {})

        if type(compare_data) == dict:
            fig, ax, c = self.plot_comparison_dict(xdata, compare_data, **kw)

        else:
            fig, ax, c = self.plot_comparison_hexbin(xdata, compare_data,
                                                     cbar, stats, **kw)

        # Aesthetics
        ax = fp.plot_one_to_one(ax)
        ax = fp.add_labels(ax, xlabel, ylabel, **label_kwargs)
        ax = fp.add_title(ax, title, **title_kwargs)

        # Make sure we have the same ticks
        # ax.set_yticks(ax.get_xticks(minor=False), minor=False)
        ax.set_xticks(ax.get_yticks(minor=False), minor=False)
        ax.set_xlim(ax.get_ylim())

        return fig, ax, c

### SPLIT REDUCED RANK JACOBIAN AND REDUCED DIMENSION JACOBIAN INTO
### DIFFERENT CLASSES
class ReducedRankJacobian(ReducedRankInversion):
    def __init__(self, k, xa, sa_vec, y, y_base, so_vec):
        # Inherit from the parent class.
        ReducedRankInversion.__init__(self, k, xa, sa_vec, y, y_base, so_vec)

        self.perturbed_cells = np.array([])
        self.model_runs = 0


    #######################################
    ### REDUCED DIMENSION JACOBIAN FUNCTIONS ###
    #######################################
    # def create_aggregate_cells_kmeans():

    def get_neighboring_cells(self, nsv_index, clusters_plot, ring=1):
        latlon = clusters_plot.where(clusters_plot == nsv_index, drop=True)
        ulim_lon = latlon.lon.values + ring*self.lonres
        llim_lon = latlon.lon.values - ring*self.lonres
        ulim_lat = latlon.lat.values + ring*self.latres
        llim_lat = latlon.lat.values - ring*self.latres
        neighboring_cells = clusters_plot.where((clusters_plot.lon <= ulim_lon) &
                                                (clusters_plot.lon >= llim_lon) &
                                                (clusters_plot.lat <= ulim_lat) &
                                                (clusters_plot.lat >= llim_lat),
                                                drop=True).values.flatten()
        return neighboring_cells[neighboring_cells > 0].astype(int)

    def get_adjacent_cells(self, nsv_index, clusters_plot, ring=1):
        latlon = clusters_plot.where(clusters_plot == nsv_index, drop=True)
        ulim_lon = latlon.lon.values + ring*self.lonres
        llim_lon = latlon.lon.values - ring*self.lonres
        ulim_lat = latlon.lat.values + ring*self.latres
        llim_lat = latlon.lat.values - ring*self.latres
        cond_lon = (clusters_plot.lon <= ulim_lon) & \
                   (clusters_plot.lon >= llim_lon) & \
                   (clusters_plot.lat == latlon.lat.values)
        cond_lat = (clusters_plot.lat <= ulim_lat) & \
                   (clusters_plot.lat >= llim_lat) & \
                   (clusters_plot.lon == latlon.lon.values)
        cond = xr.ufuncs.logical_or(cond_lon, cond_lat)
        adjacent_cells = clusters_plot.where(cond,
                                             drop=True).values.flatten()
        adjacent_cells = adjacent_cells[adjacent_cells != nsv_index]

        # get rid of kitty corner nans
        adjacent_cells = adjacent_cells[~np.isnan(adjacent_cells)]

        # get rid of 0s
        adjacent_cells = adjacent_cells[adjacent_cells > 0]

        return adjacent_cells.astype(int)

    def get_lat_adjacent_cells(self, nsv_index, clusters_plot, ring=1):
        latlon = clusters_plot.where(clusters_plot == nsv_index, drop=True)
        ulim_lat = latlon.lat.values + ring*self.latres
        llim_lat = latlon.lat.values - ring*self.latres
        cond_lat = (clusters_plot.lat <= ulim_lat) & \
                   (clusters_plot.lat >= llim_lat) & \
                   (clusters_plot.lon == latlon.lon.values)
        adjacent_cells = clusters_plot.where(cond_lat,
                                             drop=True).values.flatten()
        adjacent_cells = adjacent_cells[adjacent_cells != nsv_index]
        return adjacent_cells[~np.isnan(adjacent_cells)].astype(int)

    def get_lon_adjacent_cells(self, nsv_index, clusters_plot, ring=1):
        latlon = clusters_plot.where(clusters_plot == nsv_index, drop=True)
        ulim_lon = latlon.lon.values + ring*self.lonres
        llim_lon = latlon.lon.values - ring*self.lonres
        cond_lon = (clusters_plot.lon <= ulim_lon) & \
                   (clusters_plot.lon >= llim_lon) & \
                   (clusters_plot.lat == latlon.lat.values)
        adjacent_cells = clusters_plot.where(cond_lon,
                                             drop=True).values.flatten()
        adjacent_cells = adjacent_cells[adjacent_cells != nsv_index]
        return adjacent_cells[~np.isnan(adjacent_cells)].astype(int)

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

    @staticmethod
    def calculate_perturbation_matrix(state_vector, significance):
        sv_sig = pd.DataFrame({'sv' : state_vector,
                               'sig' : significance})
        sv_sig = sv_sig.groupby(by='sv').mean().reset_index()
        sv_sig = sv_sig.sort_values(by='sig')
        sv_sig = sv_sig[sv_sig['sv'] > 0]
        p = np.zeros((len(state_vector), sv_sig.shape[0]))
        for i, sv in enumerate(sv_sig['sv']):
            index = np.argwhere(state_vector == sv).reshape(-1,)
            p[index, i] = 0.5
        return p

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

    def calculate_k_ms(self, forward_model):
        k_ms = pd.DataFrame(forward_model)
        k_ms = k_ms.groupby(self.state_vector, axis=1).sum()
        self.k = np.array(k_ms)

    def calculate_prior_ms(self, xa_abs, sa_vec):
        xa_abs_ms = pd.DataFrame(xa_abs).groupby(self.state_vector).sum()

        sa_vec_abs = pd.DataFrame(sa_vec*xa_abs)
        sa_vec_abs_ms = (sa_vec_abs**2).groupby(self.state_vector).sum()**0.5
        sa_vec_ms = sa_vec_abs_ms/xa_abs_ms

        self.xa_abs = np.array(xa_abs_ms).reshape(-1,)
        self.xa = np.ones(self.nstate).reshape(-1,)
        self.sa_vec = np.array(sa_vec_ms).reshape(-1,)

    def calculate_trans_ms(self, nstate_native):
        gamma = np.zeros((self.nstate, nstate_native))
        for i in range(self.nstate):
            cidx = np.where(self.nstate == i )[0]
            gamma[cidx] = 1
        try:
            gamma_star = (self.sa_vec*gamma.T @
                          inv((gamma @ (self.sa_vec*gamma.T))))
            return gamma, gamma_star
        except np.linalg.LinAlgError:
            print('Singular matrix.')
            return gamma

    def calculate_significance(self,
                               pct_of_info=None, rank=None, prolongation=None):
        if prolongation is None:
            rank, prolongation, _, _ = self.projection(pct_of_info=pct_of_info,
                                                       rank=rank)
        significance = np.sqrt((prolongation**2)).sum(axis=1)

        return rank, significance

    def update_jacobian_rd(self, forward_model, xa_abs, sa_vec, clusters_plot,
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
                                  sa_vec=copy.deepcopy(self.sa_vec),
                                  y=copy.deepcopy(self.y),
                                  y_base=copy.deepcopy(self.y_base),
                                  so_vec=copy.deepcopy(self.so_vec))
        new.state_vector = copy.deepcopy(self.state_vector)
        new.dofs = copy.deepcopy(self.dofs)
        new.model_runs = copy.deepcopy(self.model_runs)
        new.rf = copy.deepcopy(self.rf)
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
        new.calculate_k_ms(forward_model)
        new.calculate_prior_ms(xa_abs=xa_abs, sa_vec=sa_vec)

        # Adjust the rf
        new.rf = self.rf*new.nstate/self.nstate

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

    # Rewrite to take any prolongation matrix?
    def update_jacobian(self, forward_model,
                        pct_of_info=None, rank=None, snr=None,
                        prolongation=None, reduction=None):#,
                        # k_base=None,
                        # convergence_threshold=1e-4, max_iterations=1000):
        # We start by creating a new instance of the class.
        # if k_base is None:
        #     k_base = copy.deepcopy(self.k)

        # Retrieve the prolongation operator associated with
        # this instance of the Jacobian for the rank specified
        # by the function call. These are the eigenvectors
        # that we will perturb.
        # if (pct_of_info is not None) and (rank is not None):
        if sum(x is not None for x in [pct_of_info, rank, snr, prolongation]) > 1:
            raise AttributeError('Conflicting arguments provided to determine rank.')
        elif sum(x is not None for x in [pct_of_info, rank, snr, prolongation]) == 0:
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
                                  sa_vec=copy.deepcopy(self.sa_vec),
                                  y=copy.deepcopy(self.y),
                                  y_base=copy.deepcopy(self.y_base),
                                  so_vec=copy.deepcopy(self.so_vec))
        new.model_runs = copy.deepcopy(self.model_runs) + prolongation.shape[1]
        new.rf = copy.deepcopy(self.rf)

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

    def plot_comparison_inversion(self, true):
        # Add variance as an object of each object
        true.shat_diag = np.diag(true.shat)
        self.shat_diag = np.diag(self.shat)

        # # set the font sizes differently
        # title_kwargs = {'fontsize' : config.SUBTITLE_FONTSIZE*config.SCALE,
        #                 'y' : 1.05}
        label_kwargs = {'labelpad' : config.LABEL_PAD/2}

        fig, ax = fp.get_figax(rows=2, cols=2)

        # Jacobian
        # title = r'Jacobian Matrix\nElements',
        fig_kwargs = {'figax' : [fig, ax[0,0]]}
        fig, ax[0,0], c = true.plot_comparison('k', self.k, cbar=False,
                                               title='Jacobian Matrix\nElements',
                                               xlabel='',
                                               ylabel='Reduced Rank',
                                               fig_kwargs=fig_kwargs,
                                               # title_kwargs=title_kwargs,
                                               label_kwargs=label_kwargs)
        # title = r'Posterior Emission\nScaling Factors'
        fig_kwargs = {'figax' : [fig, ax[0,1]]}
        fig, ax[0,1], c = true.plot_comparison('xhat', self.xhat, cbar=False,
                                                title='Posterior\nScaling Factors',
                                                xlabel='', ylabel='',
                                               fig_kwargs=fig_kwargs,
                                               # title_kwargs=title_kwargs,
                                               label_kwargs=label_kwargs)
        # title = r'Posterior Variance'
        fig_kwargs = {'figax' : [fig, ax[1,0]]}
        true.sd = np.sqrt(true.shat_diag)
        self.sd = np.sqrt(self.shat_diag)
        fig, ax[1,0], c = true.plot_comparison('sd', self.sd,
                                               cbar=False,
                                               title='Posterior Error\nStandard Deviations',
                                               ylabel='Reduced Rank',
                                               fig_kwargs=fig_kwargs,
                                               # title_kwargs=title_kwargs,
                                               label_kwargs=label_kwargs)

        # title = r'Averaging Kernel\nSensitivities'
        fig_kwargs={'figax' : [fig, ax[1,1]]}
        fig, ax[1,1], c = true.plot_comparison('dofs', self.dofs, cbar=False,
                                               title='Averaging Kernel\nSensitivities',
                                               ylabel='',
                                               fig_kwargs=fig_kwargs,
                                               # title_kwargs=title_kwargs,
                                               label_kwargs=label_kwargs)

        cax = fp.add_cax(fig, ax)
        cbar = fig.colorbar(c, cax=cax)
        cbar = fp.format_cbar(cbar, **{'cbar_title' : 'Count'})

        plt.subplots_adjust(hspace=0.5)

        return fig, ax

    def full_analysis(self, true, clusters_plot):
        # Compare inverse quantities
        fig01, ax = self.plot_comparison_inversion(true)

        # Compare spectra
        fig02, ax = true.plot_info_frac(label='Native Resolution',
                                        color=fp.color(0),
                                        text=False)
        fig02, ax = self.plot_info_frac(fig_kwargs={'figax' : [fig02, ax]},
                                        label='Update',
                                        ls=':',
                                        color=fp.color(5))

        # Plot the first few eigenvectors to give an idea of the
        # eigenspace
        rows = 2
        cols = 4
        plot_data = [('evecs', i) for i in range(cols)]

        kw = {'vmin' : -0.1,
              'vmax' : 0.1,
              'cmap' : 'RdBu_r',
              'add_colorbar' : False,
              'title_kwargs' : {'fontsize' : config.SUBTITLE_FONTSIZE*config.SCALE},
              'map_kwargs' : {'draw_labels' : False}}
        cbar_kwargs = {'ticks' : [-0.1, 0, 0.1]}

        fig03, ax = fp.get_figax(rows, cols, maps=True,
                                 lats=clusters_plot.lat,
                                 lons=clusters_plot.lon)

        kw['fig_kwargs'] = {'figax' : [fig03, ax[0, :]]}
        kw['titles'] = ['Eigenvector %d' % (i+1) for i in range(cols)]
        fig03, ax[0, :], c = true.plot_state_grid(plot_data, rows=1, cols=cols,
                                                  clusters_plot=clusters_plot,
                                                  cbar=False, **kw)

        kw['fig_kwargs'] = {'figax' : [fig03, ax[1, :]]}
        kw['titles'] = ['' for i in range(cols)]
        fig03, ax[1, :], c = self.plot_state_grid(plot_data, rows=1, cols=cols,
                                                  clusters_plot=clusters_plot,
                                                  cbar=False, **kw)

        cax = fp.add_cax(fig03, ax)
        cbar = fig03.colorbar(c, cax=cax, **cbar_kwargs)
        cbar = fp.format_cbar(cbar, **{'cbar_title' : 'Eigenvector Value'})

        # Add label
        ax[0, 0].text(-0.3, 0.5, 'Native\nResolution',
                      fontsize=config.LABEL_FONTSIZE*config.SCALE,
                      rotation=90, ha='center', va='center',
                      transform=ax[0,0].transAxes)
        ax[1, 0].text(-0.3, 0.5, 'Estimate',
                      fontsize=config.LABEL_FONTSIZE*config.SCALE,
                      rotation=90, ha='center', va='center',
                      transform=ax[1,0].transAxes)

        # # fig04
        # cbar_kwargs = {'ticks' : np.arange(-1, 4, 1),
        #                'title' : 'Scaling Factors'}
        # fig04, ax, c = self.plot_state('xhat',
        #                                clusters_plot,
        #                                default_value=1,
        #                                **{'title' : 'Posterior Emissions',
        #                                   'cmap' : 'RdBu_r',
        #                                   'vmin' : -1,
        #                                   'vmax' : 3,
        #                                   'cbar_kwargs' : cbar_kwargs})

        # # fig05
        # cbar_kwargs = {'title' : r'$\partial\hat{x}/\partial x$'}
        # fig05, ax, c = self.plot_state('dofs', clusters_plot,
        #                                **{'title' : 'Averaging Kernel',
        #                                   'cmap' : fp.cmap_trans('plasma'),
        #                                   'vmin' : 0,
        #                                   'vmax' : 0.1,#1,
        #                                   'cbar_kwargs' : cbar_kwargs})
        # ax.text(0.025, 0.05, 'DOFS = %.2f' % np.trace(self.a),
        #         fontsize=config.LABEL_FONTSIZE*config.SCALE,
        #         transform=ax.transAxes)

        return fig01, fig02, fig03 #, fig04, fig05

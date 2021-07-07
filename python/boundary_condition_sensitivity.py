import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Custom packages
import sys
sys.path.append('.')
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import format_plots as fp

np.set_printoptions(precision=3, linewidth=150)

## -------------------------------------------------------------------------##
# Define the model parameters
## -------------------------------------------------------------------------##
# Seed the random number generator
from numpy.random import RandomState
rs = RandomState(717)

# Define the parameters of our simple forward model
U = 5/3600 # windspeed (5 km/hr in km/s)
L = 25 # grid cell length (25 km)
j = U/L # transfer coeficient (s-1)
tau = 1/j
print('Lifetime (hrs): ', tau/3600)

# Dimensions of the inversion quantities
nstate = 20 # state vector
nobs_per_cell = 30
nobs = nobs_per_cell*nstate # observation vector

# Define the times at which we sample the forward model
t = np.arange(0, (nobs_per_cell+1)*0.9*L/U, 0.9*L/U)
print('Times (hrs): ', t/3600)

# Define the true and prior emissions, including the boundary conditions
BC_true = 1900 # ppb
x_true = 100*np.ones(nstate)/(3600*24) # true (ppb/s)
x_a = rs.normal(loc=80, scale=30,
                size=(nstate,))/(3600*24) # prior (ppb/s)
print('Emissions (ppb/hr):')
print('  True:  ', x_true*3600)
print('  Prior: ', x_a*3600)

# Define the prior error
s_a = 0.25*np.identity(nstate)

# Define steady state concentrations
# These will be both our initial condition and serve as the starting
# point for our pseudo-observations
y_ss = [BC_true + x_true[0]/j]
for i in range(1, nstate):
    y_ss.append(y_ss[-1] + x_true[i]/j)
y_ss = np.array(y_ss)
print('Steady state concentrations (ppb):')
print(y_ss, '\n')

# Initial conditions for the model
y_init = copy.deepcopy(y_ss)

# Create pseudo-observations
random_noise = rs.normal(-3.4, 5.6, (nstate, nobs_per_cell))
y = y_ss.reshape(-1,1) + random_noise
# y = y.flatten()

# Define the observational errror
s_o = 4**2*np.identity(nobs)

## -------------------------------------------------------------------------##
# Define forward model and inversion functions
## -------------------------------------------------------------------------##
# BC needs to be input as k*BC
def forward_model(x, y_init, BC, J, time):
    '''
    A function that calculates the mass in each reservoir
    after a given time given the following:
        x       :    vector of emissions
        y_init  :    initial atmospheric condition
        BC      :    boundary condition
        J       :    matrix of first-order, time-invariant
                     rate coefficients kiJ describing the
                     flow from reservoir i to reservoir J
        time    :    time to run the forward model (either a
                     scalar or a vector)    '''
    # Define the source (rows correspond to the reservoir
    # and columns to the times)
    s = copy.deepcopy(x).reshape((-1, 1))

    # Add the contribution from the boundary condition
    if isinstance(BC, np.ndarray):
        assert len(BC) == len(time), 'Varying BC must match time dimension.'
        s = np.tile(s, len(time))
        s[0, :] = s[0, :] + BC.reshape((1, -1))
    else:
        s[0] += BC
        s = s.reshape((-1, 1))

    # Conduct eigendecomposition of J
    w, v = np.linalg.eig(J)
    w = w.reshape((-1, 1))

    # The rows will correspond to each reservoir
    # and the columns will correspond to each time.
    # Check these dimensions now.
    if ~isinstance(time, np.ndarray):
        time = np.array([time]).reshape((1, -1))
    y_init = y_init.reshape((-1, 1))
    # s = s.reshape((-1, 1))

    # dy/dt = Jy + x + BC is not a homogeneous differential
    # equation. We calculate a particular and a homogeneous
    # solution. The particular solution will be of the form
    # m(t) = a*t + b:
    a = np.linalg.solve(J, np.zeros(y_init.shape))
    b = np.linalg.solve(J, a - s)
    y_partic = a*time + b
    # Note: if the source is 0, mass_partic is also 0

    # Then, the homogenous solution will be calculated
    # following Amos et al. (2013)
    alphas = np.linalg.solve(v, y_init - b)
    y_homog = v @ (alphas*np.exp(w*time))

    # The final mass is the sum of the two solutions
    return y_homog + y_partic

def forward_model_lw(x, y_init, BC, times, U, L):
    '''
    A function that calculates the mass in each reservoir
    after a given time given the following:
        x       :    vector of emissions (ppb/s)
        y_init  :    initial atmospheric condition
        BC      :    boundary condition
        delta_time :    time step
        U       :    wind speed
        L.      :    length scale for each grid box
    '''
    # Create an empty array (time x grid box) for the output
    ys = np.zeros((len(y_init), len(times)))
    ys[:, 0] = y_init

    # Iterate through the time steps
    delta_t = np.diff(times)
    for i, dt in enumerate(delta_t):
        try:
            bc = BC[i]
        except:
            bc = BC
        y_new = do_emissions(x, ys[:, i], dt)
        ys[:, i+1] = do_advection(x, y_new, bc, dt, U, L)
    return ys[:, 1:]

def do_emissions(x, y_prev, delta_t):
    y_new = y_prev + x*delta_t
    return y_new

def do_advection(x, y_prev, BC, delta_t, U, L):
    '''
    Advection following the Lax-Wendroff scheme
    '''
    # Calculate the courant number
    C = U*delta_t/L

    # Append the boundary conditions
    try:
        LBC = BC[0]
        RBC = BC[1]
    except:
        LBC = BC
        RBC = BC
    y_prev = np.append(LBC, y_prev)
    y_prev = np.append(y_prev, RBC)

    # Calculate the next time step
    y_new = (y_prev[1:-1]
             - C*(y_prev[2:] - y_prev[:-2])/2
             + C**2*(y_prev[2:] - 2*y_prev[1:-1] + y_prev[:-2])/2)

    return y_new

def build_jacobian(x_a, y_init, BC, times, U, L):
    F = lambda x : forward_model_lw(x=x, y_init=y_init, BC=BC,
                                    times=times, U=U, L=L).flatten()

    # Calculate prior observations
    y_a = F(x_a)

    # Iterate through the state vector elements
    K = np.zeros((len(y_a), len(x_a)))
    for i in range(len(x_a)):
        # Apply the perturbation to the ith state vector element
        x = copy.deepcopy(x_a)
        x[i] *= 1.5

        # Run the forward model
        y_pert = F(x)

        # Save out the result
        K[:, i] = (y_pert - y_a)/0.5

    return K

def solve_inversion(x_a, s_a, y, y_a, s_o, k):
    x_a = np.ones(nstate)
    c = y_a - k @ x_a
    s_hat = np.linalg.inv(np.linalg.inv(s_a) + k.T @ np.linalg.inv(s_o) @ k)
    a = np.identity(nstate) - s_hat @ np.linalg.inv(s_a)
    x_hat = (x_a
             + s_hat @ k.T @ np.linalg.inv(s_o) @ (y - k @ x_a - c))
    return x_hat, s_hat, a

def plot_inversion(x_a, x_hat, x_true):
    fig, ax = format_plot(nstate)

    xp = np.arange(1, nstate+1)
    ax.plot(xp, 3600*24*x_a, c=fp.color(5), marker='.', markersize=10,
            label='Prior')
    ax.plot(xp, 3600*24*x_a*x_hat, c=fp.color(8), marker='*', markersize=10,
               label='Posterior')
    ax.plot(xp, 3600*24*x_true, c=fp.color(3), ls='--', label='Truth')
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, 1.05), loc='lower center',
                       ncol=3)
    ax = fp.add_labels(ax, 'State Vector Element', 'Emissions (ppb/day)')
    ax.set_ylim(0, 200)

    return fig, ax

def plot_obs(nstate, y, y_a, y_ss):
    # Plot observations
    fig, ax = format_plot(nstate)

    xp = np.arange(1, nstate+1)
    ax.plot(xp, y_a, c=fp.color(3), lw=1, ls=':',
            label='Modeled')
    ax.plot(xp, y, c=fp.color(5), lw=1, marker='.', markersize=10,
            label='Observed')
    ax.plot(xp, y_ss, c=fp.color(8), lw=3,
            label='Steady State')
    # ax = fp.add_legend(ax, bbox_to_anchor=(0.5, 1.05), loc='lower center',
    #                    ncol=3)
    ax = fp.add_labels(ax, 'State Vector Element', 'XCH4 (ppb)')
    ax.set_ylim(1800, 2100)

    return fig, ax

def format_plot(nstate):
    fig, ax = fp.get_figax(aspect=3)
    for i in range(nstate+2):
        ax.axvline(i-0.5, c=fp.color(1), alpha=0.2, ls=':')
    ax.set_xlim(0.5, nstate+0.5)
    return fig, ax

## -------------------------------------------------------------------------##
# Solve the inversion with a variety of boundary conditions
## -------------------------------------------------------------------------##
# Test 1: BC = constant (1900)
BC_a = BC_true
y_a = forward_model_lw(x_a, y_init, BC_a, t, U, L)#.flatten()
K = build_jacobian(x_a, y_init, BC_a, t, U, L)
x_hat, s_hat, a = solve_inversion(x_a, s_a, y.flatten(), y_a.flatten(), s_o, K)

fig, ax = plot_inversion(x_a, x_hat, x_true)
fp.save_fig(fig, '../plots/', f'constant_BC_1900_n{nstate}_m{nobs}')
fig, ax = plot_obs(nstate, y, y_a, y_ss)
fp.save_fig(fig, '../plots', f'constant_BC_1900_n{nstate}_m{nobs}_obs')

# Test 2: constant BC = BC_true - 200 ppb
BC_a = BC_true - 200
K = build_jacobian(x_a, y_init, BC_a, t, U, L)
y_a = forward_model_lw(x_a, y_init, BC_a, t, U, L)
x_hat, s_hat, a = solve_inversion(x_a, s_a, y.flatten(), y_a.flatten(), s_o, K)

fig, ax = plot_inversion(x_a, x_hat, x_true)
fp.save_fig(fig, '../plots/', f'constant_BC_1700_n{nstate}_m{nobs}')
fig, ax = plot_obs(nstate, y, y_a, y_ss)
fp.save_fig(fig, '../plots', f'constant_BC_1700_n{nstate}_m{nobs}_obs')


# Test 3: constant BC = BC_true + 200 ppb
BC_a = BC_true + 200
K = build_jacobian(x_a, y_init, BC_a, t, U, L)
y_a = forward_model_lw(x_a, y_init, BC_a, t, U, L)
x_hat, s_hat, a = solve_inversion(x_a, s_a, y.flatten(), y_a.flatten(), s_o, K)

fig, ax = plot_inversion(x_a, x_hat, x_true)
fp.save_fig(fig, '../plots/', f'constant_BC_2100_n{nstate}_m{nobs}')
fig, ax = plot_obs(nstate, y, y_a, y_ss)
fp.save_fig(fig, '../plots', f'constant_BC_2100_n{nstate}_m{nobs}_obs')


# Test 4: BC = BC_true
BC_a = [BC_true, y_ss[-1]]
y_a = forward_model_lw(x_a, y_init, BC_a, t, U, L)#.flatten()
K = build_jacobian(x_a, y_init, BC_a, t, U, L)
x_hat, s_hat, a = solve_inversion(x_a, s_a, y.flatten(), y_a.flatten(), s_o, K)

fig, ax = plot_inversion(x_a, x_hat, x_true)
fp.save_fig(fig, '../plots/', f'true_BC_n{nstate}_m{nobs}')
fig, ax = plot_obs(nstate, y, y_a, y_ss)
fp.save_fig(fig, '../plots', f'true_BC_n{nstate}_m{nobs}_obs')

# Test 5: Oscillating BC
BC_a = 2000 - 200*np.sin(2*np.pi/t.max()*t)
K = build_jacobian(x_a, y_init, BC_a, t, U, L)
y_a = forward_model_lw(x_a, y_init, BC_a, t, U, L)
x_hat, s_hat, a = solve_inversion(x_a, s_a, y.flatten(), y_a.flatten(), s_o, K)

fig, ax = plot_inversion(x_a, x_hat, x_true)
fp.save_fig(fig, '../plots/', f'oscillating_BC_n{nstate}_m{nobs}')
fig, ax = plot_obs(nstate, y, y_a, y_ss)
fp.save_fig(fig, '../plots', f'oscillating_BC_n{nstate}_m{nobs}_obs')



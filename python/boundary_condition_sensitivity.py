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
nstate = 6 # state vector
nobs_per_cell = 10
nobs = nobs_per_cell*nstate # observation vector

# Define the times at which we sample the forward model
t = np.arange(0, nobs_per_cell*0.9*L/U, 0.9*L/U)
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

# Define true boundary condition (ppb)
BC_true = 1900

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

# Define the forward model
# Matrix of transfer coefficients
J = np.diag(j*np.ones(nstate-1), -1)
np.fill_diagonal(J, -j)

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
        y_new = do_emissions(x, ys[:, i], dt)
        ys[:, i+1] = do_advection(x, y_new, BC, dt, U, L)
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
    y_prev = np.append(BC, y_prev)
    y_prev = np.append(y_prev, BC)
    print(y_prev)

    # Calculate the next time step
    y_new = (y_prev[1:-1]
             - C*(y_prev[2:] - y_prev[:-2])/2
             + C**2*(y_prev[2:] - 2*y_prev[1:-1] + y_prev[:-2])/2)

    return y_new

def build_jacobian(x_a, y_init, BC, J, time):
    F = lambda x : forward_model(x=x, y_init=y_init, BC=BC,
                                 J=J, time=time).flatten()

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

## -------------------------------------------------------------------------##
# Solve the inversion with a variety of boundary conditions
## -------------------------------------------------------------------------##
# Test 1: BC = BC_true
BC_a = BC_true
BC_a_input = BC_a*j
# K = build_jacobian(x_a, y_init, BC_a_input, J, t)
# x, y_init, BC, times, U, L
y_a = forward_model_lw(x_a, y_init, BC_a, t, U, L)#.flatten()
# print(y_a)
# print(y_a.shape)
# x_hat, s_hat, a = solve_inversion(x_a, s_a, y.flatten(), y_a, s_o, K)
# print(x_a*x_hat)
# print(x_true)

xp = np.arange(1, nstate+1)
fig, ax = fp.get_figax(aspect=2)
# ax.set_frame_on(False)
for i in range(8):
    ax.axvline(i-0.5, c=fp.color(1), alpha=0.2, ls=':')

ax.plot(xp, y_a, c=fp.color(3),
        label='Modeled')
ax.plot(xp, y, c=fp.color(5), marker='.', markersize=10,
        label='Observed')
ax.plot(xp, y_ss, c=fp.color(8),
        label='Steady State')

ax = fp.add_legend(ax, bbox_to_anchor=(0.5, 1.05), loc='lower center',
                   ncol=3)
ax = fp.add_labels(ax, 'State Vector Element', 'XCH4 (ppb)')
ax.set_xlim(0.5, 6.5)
ax.set_ylim(1800, 2100)

plt.show()

# # Test 2: constant BC = BC_true - 200 ppb
# BC_a = BC_true - 200
# BC_a_input = BC_a*j
# K = build_jacobian(x_a, y_init, BC_a_input, J, t)
# y_a = forward_model(x_a, y_init, BC_a_input, J, t).flatten()
# x_hat_minus, s_hat_minus, a_minus = solve_inversion(x_a, s_a, y, y_a, s_o, K)

# # Test 3: constant BC = BC_true + 200 ppb
# BC_a = BC_true + 200
# BC_a_input = BC_a*j
# K = build_jacobian(x_a, y_init, BC_a_input, J, t)
# y_a = forward_model(x_a, y_init, BC_a_input, J, t).flatten()
# x_hat_plus, s_hat_plus, a_plus = solve_inversion(x_a, s_a, y, y_a, s_o, K)

# # Test 4: Oscillating BC
# BC_a = 2000 + 200*np.sin(2*np.pi/t.max()*t)
# BC_a_input = BC_a*j # Adjust it to reflect inflow
# K = build_jacobian(x_a, y_init, BC_a_input, J, t)
# y_a = forward_model(x_a, y_init, BC_a_input, J, t).flatten()
# x_hat_osc, s_hat_osc, a_osc = solve_inversion(x_a, s_a, y, y_a, s_o, K)

# ## -------------------------------------------------------------------------##
# # Plot some things
# ## -------------------------------------------------------------------------##
# # Scale emissions
# # x_true *= 3600*24
# # x_a *= 3600*24
# # x_hat *= 3600*24
# # x_hat_minus *= 3600*24
# # x_hat_plus *= 3600*24
# # x_hat_osc *= 3600*24

# xp = np.arange(1, nstate+1)
# fig, ax = fp.get_figax(aspect=2)
# # ax.set_frame_on(False)
# for i in range(8):
#     ax.axvline(i-0.5, c=fp.color(1), alpha=0.2, ls=':')
# # ax.fill_between(xp, x_hats[0, :], x_hats[-1, :],
# #                    color=fp.color(5), alpha=0.3,
# #                    label='BC = 2000 +/- 200 ppb')
# # ax.plot(xp, x_hats[2, :], color=fp.color(5), lw=2,
# #         label='BC = 1900 ppb')
# ax.scatter(xp, x_true, c=fp.color(1), marker='*',
#               s=50, label='True Emissions (ppb/day)', zorder=1)
# ax.scatter(xp, x_a, c=fp.color(5), marker='*',
#               s=50, label='Prior Emissions (ppb/day)', zorder=1)
# ax.scatter(xp, x_hat*x_a)


# # ax2 = ax.twinx()
# # ax2.scatter(xp, y_true, c=fp.color(5), s=50)
# # ax2.set_ylim(1880, 2020)
# # ax2.set_ylabel('Steady State Concentration (ppb)', color=fp.color(5))
# # ax2.tick_params(axis='y', labelcolor=fp.color(5))
# # ax2.set_facecolor('white')

# ax = fp.add_legend(ax, bbox_to_anchor=(0.5, 1.05), loc='lower center',
#                    ncol=2)
# ax = fp.add_labels(ax, 'State Vector Element', 'Emissions (ppb/day)')
# ax.set_xlim(0.5, 6.5)

# # ax[0].get_shared_x_axes().join(ax[0], a[1])
# # ax[0].set_xticklabels([])
# # ax[0].set_frame_on(False)
# # ax[0].set_xticks([])

# # ax[0].scatter(xp, y_true, c=fp.color(1), marker='.', s=50)
# # ax[0].axhline(1900, color=fp.color(5), lw=2)

# # ax[0] = fp.add_labels(ax[0], '', 'Steady State\nConcentration (ppb)')
# # ax[0].set_ylim(1875, 2150)

# # fp.save_fig(fig, '../plots/', 'bc_test_02')

# plt.show()


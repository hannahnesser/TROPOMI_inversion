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

# NOTE: This is all hard coded to be a relative inversion.

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
plot_dir = '../plots/BC_sensitivity'

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
nstate = 30 # state vector
nobs_per_cell = 30
nobs = nobs_per_cell*nstate # observation vector

# Define the times at which we sample the forward model
C = 0.5 # Courant number
delta_t = C*L/U
t = np.arange(0, (nobs_per_cell+1)*delta_t, delta_t)
print('Times (hrs): ', t/3600)

# Define the true and prior emissions, including the boundary conditions
BC_true = 1900 # ppb
x_true = 100*np.ones(nstate)/(3600*24) # true (ppb/s)
x_a = rs.normal(loc=80, scale=30,
                size=(nstate,))/(3600*24) # prior (ppb/s)
# x_a[x_a*3600*24 < 50] = 50/(3600*24)
print('Emissions (ppb/hr):')
print('  True:  ', x_true*3600)
print('  Prior: ', x_a*3600)

# Define the prior error (This is n+1 x n+1 to accomodate BC optimization)
s_a = 0.5**2*np.identity(nstate)
idx = x_a < 0.5*x_a.mean()
# idx = np.append(False, idx)
s_a[idx, idx] = 0.5*x_a.mean()/x_a[idx]

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
random_noise = rs.normal(0, 10, (nstate, nobs_per_cell))
y = y_ss.reshape(-1,1) + random_noise
# y = y.flatten()

# Define the observational errror
s_o = 15**2*np.identity(nobs)

## -------------------------------------------------------------------------##
# Define forward model and inversion functions
## -------------------------------------------------------------------------##
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
    # Create an empty array (grid box x time) for the output
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
        # y_new = do_advection(x, ys[:, i], bc, dt, U, L)
        # ys[:, i+1] = do_emissions(x, y_new, dt)
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

    # Calculate the next time step using Lax-Wendroff
    y_new = (y_prev[1:-1]
             - C*(y_prev[2:] - y_prev[:-2])/2
             + C**2*(y_prev[2:] - 2*y_prev[1:-1] + y_prev[:-2])/2)

    # Update the last grid cell using upstream
    y_new = np.append(y_new, y_prev[-1] - C*(y_prev[-1] - y_prev[-2]))

    return y_new

def build_jacobian(x_a, y_init, BC, times, U, L, optimize_BC=False):
    F = lambda x : forward_model_lw(x=x, y_init=y_init, BC=BC,
                                    times=times, U=U, L=L).flatten()

    # Calculate prior observations
    y_a = F(x_a)

    # Initialize the Jacobian
    K = np.zeros((len(y_a), len(x_a)))

    # Iterate through the state vector elements
    for i in range(len(x_a)):
        # Apply the perturbation to the ith state vector element
        x = copy.deepcopy(x_a)
        x[i] *= 1.5

        # Run the forward model
        y_pert = F(x)

        # Save out the result
        K[:, i] = (y_pert - y_a)/0.5

    if optimize_BC:
        # Add a column for the optimization of the boundary condition
        y_pert = forward_model_lw(x=x_a, y_init=y_init, BC=1.5*BC,
                                  times=times, U=U, L=L).flatten()
        dy_dx = ((y_pert - y_a)/0.5).reshape(-1, 1)
        K = np.append(dy_dx, K, axis=1)

    return K

def solve_inversion(x_a, s_a, y, y_a, s_o, k, optimize_BC=False):
    if optimize_BC:
        n = nstate + 1
    else:
        n = nstate

    x_a = np.ones(n)
    c = y_a - k @ x_a
    s_hat = np.linalg.inv(np.linalg.inv(s_a) + k.T @ np.linalg.inv(s_o) @ k)
    a = np.identity(n) - s_hat @ np.linalg.inv(s_a)
    x_hat = (x_a
             + s_hat @ k.T @ np.linalg.inv(s_o) @ (y - k @ x_a - c))
    return x_hat, s_hat, a

def plot_inversion(x_a, x_hat, x_true, x_hat_true=None):
    fig, ax = format_plot(nstate)

    xp = np.arange(1, nstate+1)
    ax.plot(xp, 3600*24*x_true, c=fp.color(2), ls='--', label='Truth')
    ax.plot(xp, 3600*24*x_a, c=fp.color(4), marker='.', markersize=10,
            label='Prior')
    if x_hat_true is not None:
        ax.plot(xp, 3600*24*x_hat_true*x_a, c=fp.color(6),
                marker='*', markersize=10, label='True BC Posterior')
        ax.plot(xp, 3600*24*x_a*x_hat, c=fp.color(8), marker='.', markersize=5,
                lw=1, label='Posterior')
        ncol = 2
    else: # if x_hat_true is none
        ax.plot(xp, 3600*24*x_a*x_hat, c=fp.color(6), marker='*', markersize=10,
               label='True BC Posterior')

        ncol = 3

    # Aesthetics
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.35), loc='upper center',
                       ncol=ncol)
    ax = fp.add_labels(ax, 'State Vector Element', 'Emissions (ppb/day)')
    ax.set_ylim(0, 200)

    return fig, ax

def plot_obs(nstate, y, y_a, y_ss):
    # Plot observations
    fig, ax = format_plot(nstate)

    xp = np.arange(1, nstate+1)
    ax.plot(xp, y_ss, c='black', lw=2,
            label='Steady State', zorder=10)
    ax.plot(xp, y, c='grey', lw=0.5, ls=':', markersize=10,
            label='Observed', zorder=9)
    for i, y_a_column in enumerate(y_a.T):
        if (i == 0) or (i == (len(t) - 2)):
            ax.plot(xp, y_a_column,
                    c=fp.color(i+2, cmap='plasma', lut=len(t+2)),
                    lw=0.5, ls='-', label=f'Modeled (t={(t[i+1]/3600)} hrs)')
        else:
            ax.plot(xp, y_a_column,
                    c=fp.color(i+2, cmap='plasma', lut=len(t+2)),
                    lw=0.5, ls='-')

    # Aesthetics
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.35), loc='upper center',
                       ncol=2)
    ax = fp.add_labels(ax, 'State Vector Element', 'XCH4 (ppb)')
    ax.set_ylim(1800, 2400)

    return fig, ax

def format_plot(nstate):
    fig, ax = fp.get_figax(aspect=3)
    for i in range(nstate+2):
        ax.axvline(i-0.5, c=fp.color(1), alpha=0.2, ls=':')
    ax.set_xlim(0.5, nstate+0.5)
    ax.set_facecolor('white')
    return fig, ax

## -------------------------------------------------------------------------##
# Solve the inversion with a variety of boundary conditions
## -------------------------------------------------------------------------##
# Inversion plots
xp = np.arange(1, nstate+1)

# Test 1: BC = constant (1900)
y_a = forward_model_lw(x_a, y_init, BC_true, t, U, L)#.flatten()
K = build_jacobian(x_a, y_init, BC_true, t, U, L)
x_hat_true, s_hat, a = solve_inversion(x_a, s_a,
                                       y.flatten(), y_a.flatten(), s_o, K)
# x_hat_true = x_hat_true[1:]

fig, ax = plot_inversion(x_a, x_hat_true, x_true)
ax = fp.add_title(ax, f'True Boundary Condition\n(BC = {BC_true:d} ppb)')
fp.save_fig(fig, plot_dir, f'constant_BC_{BC_true:d}_n{nstate}_m{nobs}')

fig, ax = plot_obs(nstate, y, y_a, y_ss)
ax = fp.add_title(ax, f'True Boundary Condition\n(BC = {BC_true:d} ppb)')
fp.save_fig(fig, plot_dir, f'constant_BC_{BC_true:d}_n{nstate}_m{nobs}_obs')
plt.close()

# Test 2: constant BC perturbation
perts = [50, 100, 200, 300, 400]
fig_summ, ax_summ = format_plot(nstate)
for i, pert in enumerate(perts):
    BC = BC_true + pert

    # Solve inversion
    K = build_jacobian(x_a, y_init, BC, t, U, L)
    y_a = forward_model_lw(x_a, y_init, BC, t, U, L)
    x_hat, s_hat, a = solve_inversion(x_a, s_a,
                                      y.flatten(), y_a.flatten(), s_o, K)
    # x_hat = x_hat[1:]

    fig, ax = plot_inversion(x_a, x_hat, x_true, x_hat_true)
    ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
    fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_n{nstate}_m{nobs}')

    fig, ax = plot_obs(nstate, y, y_a, y_ss)
    ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
    fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_n{nstate}_m{nobs}_obs')

    ax_summ.plot(xp, np.abs(x_hat - x_hat_true)*x_a*3600*24,
                 c=fp.color(k=2*i), lw=2, ls='-',
                 label=f'{pert:d}/-{pert:d} ppb')

fp.add_title(ax_summ, 'Constant Boundary Condition Perturbations')
fp.add_labels(ax_summ, 'State Vector Element', r'$\Delta$XCH4 (ppb)')
fp.add_legend(ax_summ, bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=3)
fp.save_fig(fig_summ, plot_dir, f'constant_BC_summary_n{nstate}_m{nobs}')
plt.close()

# Test 3: oscillating BC perturbation
# Summary plot
fig_perts, ax_perts = fp.get_figax(aspect=5)
fig_summ, ax_summ = format_plot(nstate)
BC1 = 2000 - 200*np.cos(2*np.pi/t.max()*t)
BC2 = 2000 + 200*np.cos(2*np.pi/t.max()*t)
BC3 = 2000 - 100*np.cos(2*np.pi/t.max()*t)
BC4 = 2000 - 200*np.cos(4*np.pi/t.max()*t)
BC5 = 2000 - 200*np.cos(6*np.pi/t.max()*t)
BCs = [BC1, BC2, BC3, BC4, BC5]
for i, BC in enumerate(BCs):
    # Summary plot
    ax_perts.plot(t/3600, BC, c=fp.color(i*2, lut=len(BCs)*2), lw=2)

    # Solve inversion
    K = build_jacobian(x_a, y_init, BC, t, U, L)
    y_a = forward_model_lw(x_a, y_init, BC, t, U, L)
    x_hat, s_hat, a = solve_inversion(x_a, s_a,
                                      y.flatten(), y_a.flatten(), s_o, K)
    # x_hat = x_hat[1:]

    # Plots
    fig, ax = plot_inversion(x_a, x_hat, x_true, x_hat_true)
    fp.save_fig(fig, plot_dir, f'oscillating_BC{i}_n{nstate}_m{nobs}')
    fig, ax = plot_obs(nstate, y, y_a, y_ss)
    fp.save_fig(fig, plot_dir, f'oscillating_BC{i}_n{nstate}_m{nobs}_obs')

    # Summary plot
    ax_summ.plot(xp, np.abs(x_hat - x_hat_true)*x_a*3600*24,
                 c=fp.color(i*2, lut=len(BCs)*2), lw=2, ls='-',
                 label=f'Test {i+1}')

ax_perts.set_xlim(t.min()/3600, t.max()/3600)
ax_perts.set_ylim(1600, 2300)
ax_perts = fp.add_labels(ax_perts, 'Time (hr)', 'BC (ppb)')
fp.save_fig(fig_perts, plot_dir, f'oscillating_BC_perts_summary')

fp.add_title(ax_summ, 'Oscillating Boundary Condition Perturbations')
fp.add_labels(ax_summ, 'State Vector Element', r'$\Delta$XCH4 (ppb)')
fp.add_legend(ax_summ, bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=3)
fp.save_fig(fig_summ, plot_dir, f'oscillating_BC_summary_n{nstate}_m{nobs}')
plt.close()

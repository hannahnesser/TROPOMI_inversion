import numpy as np
from numpy.linalg import inv
# from scipy.linalg import inv
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Custom packages
import sys
sys.path.append('.')
import gcpy as gc
import config
config.SCALE = config.PRES_SCALE
config.BASE_WIDTH = config.PRES_WIDTH
config.BASE_HEIGHT = config.PRES_HEIGHT
import format_plots as fp

np.set_printoptions(precision=3, linewidth=300, suppress=True)

# NOTE: This is all hard coded to be a relative inversion.

## -------------------------------------------------------------------------##
# File Locations
## -------------------------------------------------------------------------##
plot_dir = '../plots/BC_sensitivity'

## -------------------------------------------------------------------------##
# Define the model parameters
## -------------------------------------------------------------------------##
optimize_BC = True

# Seed the random number generator
from numpy.random import RandomState
rs = RandomState(717)

# Define the parameters of our simple forward model
U = 5/3600 # windspeed (5 km/hr in km/s)
L = 25 #12.5 # grid cell length (25 km)
j = U/L # transfer coeficient (s-1)
tau = 1/j

# Dimensions of the inversion quantities
nstate = 20 #30 # state vector
nobs_per_cell = 15 #30
nobs = nobs_per_cell*nstate # observation vector

# Define the times
init_t = 150*3600
total_t = 150*3600 # time in seconds

# Define the times at which we sample the forward model and the observations
C = 0.5 # Courant number
delta_t = C*L/U # seconds
t = np.arange(0, init_t + total_t + delta_t, delta_t)
obs_t = np.linspace(init_t + delta_t, init_t + total_t, nobs_per_cell)

# Define the true emissions, including the boundary conditions
BC_true = 1900 # ppb
x_true = 100*np.ones(nstate)/(3600*24) # true (ppb/s)

# Define the prior and prior error
x_a = rs.normal(loc=80, scale=30,
                size=(nstate,))/(3600*24) # prior (ppb/s)
idx_xa = x_a < 0.5*x_a.mean()

s_a_err = 0.5*x_a.mean()/x_a
s_a_err[s_a_err < 0.5] = 0.5
# s_a_err[0] = 4
if optimize_BC:
    s_a_err = np.append(s_a_err, 0.5)
    s_a = s_a_err**2*np.identity(nstate + 1)
else:
    s_a = s_a_err**2*np.identity(nstate)

# Define steady state concentrations
# These will be both our initial condition and serve as the starting
# point for our pseudo-observations
y_ss = [BC_true + x_true[0]/j]
for i in range(1, nstate):
    y_ss.append(y_ss[-1] + x_true[i]/j)
y_ss = np.array(y_ss)

# Initial conditions for the model
y_init = copy.deepcopy(y_ss)

# Create pseudo-observations
random_noise = rs.normal(0, 10, (nstate, nobs_per_cell))
y = y_ss.reshape(-1,1) + random_noise
# y = y.flatten()

# Define the observational errror
s_o_err = 15 #15
s_o = s_o_err**2*np.identity(nobs)

## -------------------------------------------------------------------------##
# Print information about the study
## -------------------------------------------------------------------------##
print('-'*20, 'BOUNDARY CONDITION SENSITIVITY STUDY', '-'*20)
print('MODEL PARAMETERS')
print(f'GRID CELL LENGTH (km)    : {L}')
print(f'NUMBER OF GRID CELLS     : {nstate}')
print(f'WIND SPEED (km/hr)       : {(U*3600)}')
print(f'TIME STEP (hrs)          : {(delta_t/3600)}')
print(f'GRID CELL LIFETIME (hrs) : {(tau/3600)}')
print(f'TRUE EMISSIONS (ppb/hr)  :', x_true[0]*3600)
print(f'STEADY STATE (ppb)       :', y_ss)
print(f'OBSERVATION TIMES (hrs)  :', obs_t/3600)
print(f'MODEL TIMES (hrs)        :', t/3600)

print('')

print('INVERSION PARAMETERS')
print('PRIOR EMISSIONS           : ', x_a*3600)

print('-'*78)

## -------------------------------------------------------------------------##
# Define forward model and inversion functions
## -------------------------------------------------------------------------##
def forward_model_lw(x, y_init, BC, ts, U, L, obs_t):
    '''
    A function that calculates the mass in each reservoir
    after a given time given the following:
        x         :    vector of emissions (ppb/s)
        y_init    :    initial atmospheric condition
        BC        :    boundary condition
        ts        :    times at which to sample the model
        U         :    wind speed
        L         :    length scale for each grid box
        obs_t     :    times at which to sample the model
    '''
    # Create an empty array (grid box x time) for all
    # model output
    ys = np.zeros((len(y_init), len(ts)))
    ys[:, 0] = y_init

    # Iterate through the time steps
    for i, t in enumerate(ts[1:]):
        # Get boundary condition
        try:
            bc = BC[i]
        except:
            bc = BC

        # Get
        y_new = do_emissions(x, ys[:, i], delta_t)
        ys[:, i+1] = do_advection(x, y_new, bc, delta_t, U, L)

    # Subset all output for observational times
    t_idx = gc.nearest_loc(obs_t, ts)
    ys = ys[:, t_idx]

    return ys

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

def build_jacobian(x_a, y_init, BC, ts, U, L, obs_t,
                   optimize_BC=False):
    F = lambda x : forward_model_lw(x=x, y_init=y_init, BC=BC, ts=ts,
                                    U=U, L=L, obs_t=obs_t).flatten()

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
        y_pert = forward_model_lw(x=x_a, y_init=y_init, BC=1.5*BC, ts=ts,
                                  U=U, L=L, obs_t=obs_t).flatten()
        dy_dx = ((y_pert - y_a)/0.5).reshape(-1, 1)
        K = np.append(K, dy_dx, axis=1)

    return K

def solve_inversion(x_a, s_a, y, y_a, s_o, k,
                    optimize_BC=False, verbose=False):
    if optimize_BC:
        n = nstate + 1
    else:
        n = nstate

    # Get the inverse of s_a and s_o
    s_a_inv = np.diag(1/np.diag(s_a))
    s_o_inv = np.diag(1/np.diag(s_o))

    x_a = np.ones(n)
    c = y_a - k @ x_a
    s_hat = inv(s_a_inv + k.T @ s_o_inv @ k)
    g = s_hat @ k.T @ s_o_inv
    a = np.identity(n) - s_hat @ s_a_inv
    x_hat = (x_a + s_hat @ k.T @ s_o_inv @ (y - k @ x_a - c))

    if optimize_BC:
        x_hat = x_hat[:-1]
        s_hat = s_hat[:-1, :-1]
        a = a[:-1, :-1]

    if verbose:
        print('-'*10, 'c', '-'*10)
        print(c)
        # print('-'*10, 'K', '-'*10)
        # print(k)
        # print('-'*10, 'G', '-'*10)
        # print(g)
        print('-'*10, 'G SUM', '-'*10)
        print(g.sum(axis=1))

    return x_hat, s_hat, a

def calculate_gain(s_a, s_o, k):
    # Get the inverse of s_a and s_o
    s_a_inv = np.diag(1/np.diag(s_a))
    s_o_inv = np.diag(1/np.diag(s_o))
    return inv(s_a_inv + k.T @ s_o_inv @ k) @ k.T @ s_o_inv

def plot_inversion(x_a, x_hat, x_true, x_hat_true=None, s_a=None, a=None,
                   optimize_BC=False, plot_avker=False):
    # Set up plots
    if plot_avker:
        fig, axis = format_plot(nstate, nplots=2)
        ax = axis[0]
    else:
        fig, ax = format_plot(nstate, nplots=1)

    # Subset prior error
    if optimize_BC and (s_a is not None):
        s_a = s_a[:-1, :-1]

    # Add text
    add_text(ax)

    # Get plotting x coordinates
    xp = np.arange(1, nstate+1)

    # Plot "true " emissions
    ax.plot(xp, 3600*24*x_true, c=fp.color(2), ls='--', label='Truth')

    # Plot prior
    if s_a is None:
        ax.plot(xp, 3600*24*x_a, c=fp.color(4), marker='.', markersize=10,
                label='Prior')
    else:
        ax.errorbar(xp, 3600*24*x_a, yerr=3600*24*np.diag(s_a)**0.5*x_a,
                    c=fp.color(4), marker='.', markersize=10,
                    label='Prior')

    # Pot posterior
    if x_hat_true is not None:
        ax.plot(xp, 3600*24*x_hat_true*x_a, c=fp.color(6),
                marker='*', markersize=10, label='True BC Posterior')
        ax.plot(xp, 3600*24*x_a*x_hat, c=fp.color(8), marker='.', markersize=5,
                lw=1, label='Posterior')
        ncol = 2
    else: # if x_hat_true is none
        ax.plot(xp, 3600*24*x_a*x_hat, c=fp.color(6),
                marker='*', markersize=10,
                label='True BC Posterior')

        ncol = 3

    # Plot avker
    if plot_avker:
        axis[1].plot(xp, np.diag(a))
        ax = axis
    else:
        ax = [ax]

    ax[-1] = fp.add_legend(ax[-1], bbox_to_anchor=(0.5, -0.35),
                           loc='upper center', ncol=ncol)
    for a in ax:
        a = fp.add_labels(a, 'State Vector Element', 'Emissions (ppb/day)')
        a.set_ylim(0, 200)

    if plot_avker:
        return fig, ax
    else:
        return fig, ax[0]

def plot_avker(nstate, a):
    fig, ax = format_plot(nstate, nplots=1)
    add_text(ax)

    # Get plotting x coordinates
    xp = np.arange(1, nstate+1)

    # Plot avker sensitivities
    ax.plot(xp, np.diag(a), c=grey, lw=1)

    # Add text
    fp.add_labels(ax, 'State Vector Element', 'Averaging kernel\nsensitivities')

    # Set limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    return ax

def plot_obs(nstate, y, y_a, y_ss):
    # Plot observations
    fig, ax = format_plot(nstate)

    xp = np.arange(1, nstate+1)
    ax.plot(xp, y_ss, c='black', lw=2, label='Steady State', zorder=10)
    ax.plot(xp, y, c='grey', lw=0.5, ls=':', markersize=10,
            label='Observed', zorder=9)
    for i, y_a_column in enumerate(y_a.T):
        if (i == 0) or (i == (len(obs_t) - 1)):
            ax.plot(xp, y_a_column,
                    c=fp.color(i+2, cmap='plasma', lut=len(obs_t+2)),
                    lw=0.5, ls='-',
                    label=f'Modeled (t={(obs_t[i]/3600):.1f} hrs)')
        else:
            ax.plot(xp, y_a_column,
                    c=fp.color(i+2, cmap='plasma', lut=len(obs_t+2)),
                    lw=0.5, ls='-')

    # Aesthetics
    add_text(ax)
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.35), loc='upper center',
                       ncol=2)
    ax = fp.add_labels(ax, 'State Vector Element', 'XCH4 (ppb)')
    ax.set_ylim(y_ss.min()-50, y_ss.max()+50)

    return fig, ax

def format_plot(nstate, nplots=1, **fig_kwargs):
    fig, ax = fp.get_figax(aspect=4 - 0.65*(nplots-1), cols=1, rows=nplots,
                           **fig_kwargs)
    if nplots == 1:
        ax = [ax]
    for axis in ax:
        for i in range(nstate+2):
            axis.axvline(i-0.5, c=fp.color(1), alpha=0.2, ls=':')
        axis.set_xticks(np.arange(0, nstate+1, 2))
        axis.set_xlim(0.5, nstate+0.5)
        axis.set_facecolor('white')
    if nplots == 1:
        return fig, ax[0]
    else:
        return fig, ax

def add_text(ax):
    if optimize_BC:
        txt = 'BC optimized'
    else:
        txt = 'BC not optimized'
    # txt = txt + f'\nn = {nstate}\nm = {nobs}\nU = {(U*3600)}'
    ax.text(0.98, 0.95, txt, ha='right', va='top',
                 fontsize=config.LABEL_FONTSIZE*config.SCALE,
                 transform=ax.transAxes)

## -------------------------------------------------------------------------##
# Create initial plots
## -------------------------------------------------------------------------##
# Prior
fig, ax = format_plot(nstate, nplots=2, sharex=True)
fp.add_title(ax[0], 'Base inversion variables')
xp = np.arange(1, nstate+1) # plotting x coordinates

# subset prior error
if optimize_BC:
    s_a_p = s_a[:-1, :-1]
else:
    s_a_p = s_a

# Plot "true " emissions
ax[0].plot(xp, 3600*24*x_true, c=fp.color(2), ls='--', label='Truth')
ax[0].errorbar(xp, 3600*24*x_a, yerr=3600*24*np.diag(s_a_p)**0.5*x_a,
            c=fp.color(4), marker='.', markersize=10, capsize=2,
            label=r'Prior ($\pm$$\approx$ 50\%)')
handles_0, labels_0 = ax[0].get_legend_handles_labels()

ax[0] = fp.add_labels(ax[0], '', 'Emissions\n(ppb/day)')
ax[0].set_ylim(0, 200)

# Observations
ax[1].plot(xp, y_ss, c='black', label='Steady state', zorder=10)
ax[1].plot(xp, y, c='grey', label='Observations ($\pm$ 15 ppb)',
           zorder=9, alpha=0.4)

# Error range
y_err_min = (y - 15).min(axis=1)
y_err_max = (y + 15).max(axis=1)
ax[1].fill_between(xp, y_err_min, y_err_max, color='grey', alpha=0.2)
                   # label=r'Error range ($\pm$ 15 ppb)')
handles_1, labels_1 = ax[1].get_legend_handles_labels()
handles_0.extend(handles_1)
labels_0.extend(labels_1)

# Aesthetics
ax[1] = fp.add_legend(ax[1], handles=handles_0, labels=labels_0,
                      bbox_to_anchor=(0.5, -0.35),
                      loc='upper center', ncol=2)
ax[1] = fp.add_labels(ax[1], 'State Vector Element', 'XCH4\n(ppb)')
ax[1].set_ylim(y_ss.min()-50, y_ss.max()+50)

fp.save_fig(fig, plot_dir, f'prior_obs_n{nstate}_m{nobs}_obs')

## -------------------------------------------------------------------------##
# Solve the inversion with a variety of boundary conditions
## -------------------------------------------------------------------------##
# Inversion plots
xp = np.arange(1, nstate+1)

# Test 1: BC = constant (1900)
y_a = forward_model_lw(x_a, y_init, BC_true, t, U, L, obs_t)
K = build_jacobian(x_a, y_init, BC_true, t, U, L, obs_t,
                   optimize_BC)
G_sum = calculate_gain(s_a, s_o, K).sum(axis=1)

if optimize_BC:
    G_sum = G_sum[:-1]
x_hat_true, s_hat, a_true = solve_inversion(x_a, s_a,
                                            y.flatten(), y_a.flatten(), s_o, K,
                                            optimize_BC, verbose=True)

fig, ax = plot_inversion(x_a, x_hat_true, x_true, #s_a=s_a,
                        optimize_BC=optimize_BC)
ax = fp.add_title(ax, f'True Boundary Condition\n(BC = {BC_true:d} ppb)')
fp.save_fig(fig, plot_dir, f'constant_BC_{BC_true:d}_n{nstate}_m{nobs}')

fig, ax = plot_obs(nstate, y, y_a, y_ss)
ax = fp.add_title(ax, f'True Boundary Condition\n(BC = {BC_true:d} ppb)')
fp.save_fig(fig, plot_dir, f'constant_BC_{BC_true:d}_n{nstate}_m{nobs}_obs')
plt.close()

# Test 2: constant BC perturbation
perts = [50, 100, 200]
# fig_summ, ax_summ = format_plot(nstate, nplots=2, sharex=True)
fig_summ, ax_summ = format_plot(nstate)
for i, pert in enumerate(perts):
    BC = BC_true + pert

    # Solve inversion
    K = build_jacobian(x_a, y_init, BC, t, U, L, obs_t,
                       optimize_BC)
    y_a = forward_model_lw(x_a, y_init, BC, t, U, L, obs_t)
    x_hat, s_hat, a = solve_inversion(x_a, s_a,
                                      y.flatten(), y_a.flatten(), s_o, K,
                                      optimize_BC)

    fig, ax = plot_inversion(x_a, x_hat, x_true, x_hat_true,
                             optimize_BC=optimize_BC)
    ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
    fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_n{nstate}_m{nobs}')

    # fig, ax = plot_obs(nstate, y, y_a, y_ss)
    # ax = fp.add_title(ax, f'High Boundary Condition\n(BC = {BC:d} ppb)')
    # fp.save_fig(fig, plot_dir, f'constant_BC_{BC:d}_n{nstate}_m{nobs}_obs')

    # Summary plot
    # ax_summ[0].plot(xp, np.abs(x_hat - x_hat_true)*x_a*3600*24,
    #                 c=fp.color(k=4*i), lw=1, ls='-',
    #                 label=f'{pert:d}/-{pert:d} ppb')
    # ax_summ[0].plot(xp, np.abs(-pert*G_sum)*x_a*3600*24,
    #                 c=fp.color(k=4*i), lw=2, ls='--')
    # ax_summ[1].plot(xp, np.diag(a),
    #                 c='grey', lw=1, ls='-')
    ax_summ.plot(xp, np.abs(x_hat - x_hat_true)*x_a*3600*24,
                    c=fp.color(k=4*i), lw=1, ls='-',
                    label=f'{pert:d}/-{pert:d} ppb')
    ax_summ.plot(xp, np.abs(-pert*G_sum)*x_a*3600*24,
                    c=fp.color(k=4*i), lw=2, ls='--')

# # Add text
# fp.add_title(ax_summ[0], 'Constant Boundary Condition Perturbations')
# add_text(ax_summ[0])
# fp.add_labels(ax_summ[0], '', r'$\vert\Delta\hat{x}\vert$ (ppb)')
# fp.add_labels(ax_summ[1], 'State Vector Element',
#               'Averaging kernel\nsensitivities')

# # Legend for summary plot
# custom_lines = [Line2D([0], [0], color='grey', lw=1, ls='-'),
#                 Line2D([0], [0], color='grey', lw=2, ls='--')]
# custom_labels = ['Numerical solution', 'Predicted solution']
# handles, labels = ax_summ[0].get_legend_handles_labels()
# custom_lines.extend(handles)
# custom_labels.extend(labels)
# fp.add_legend(ax_summ[1], handles=custom_lines, labels=custom_labels,
#               bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=3)

# # Set limits
# ax_summ[0].set_ylim(0, 200)
# ax_summ[1].set_ylim(0, 1)
# ax_summ[1].set_yticks([0, 0.5, 1])

# Add text
fp.add_title(ax_summ, 'Constant Boundary Condition Perturbations')
add_text(ax_summ)
fp.add_labels(ax_summ, 'State Vector Element',
              r'$\vert\Delta\hat{x}\vert$ (ppb)')

# Legend for summary plot
custom_lines = [Line2D([0], [0], color='grey', lw=1, ls='-'),
                Line2D([0], [0], color='grey', lw=2, ls='--')]
custom_labels = ['Numerical solution', 'Predicted solution']
handles, labels = ax_summ.get_legend_handles_labels()
custom_lines.extend(handles)
custom_labels.extend(labels)
fp.add_legend(ax_summ, handles=custom_lines, labels=custom_labels,
              bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=3)

# Set limits
ax_summ.set_ylim(0, 200)

# Save plot
fp.save_fig(fig_summ, plot_dir, f'constant_BC_summary_n{nstate}_m{nobs}')
plt.close()

# # Test 3: oscillating BC perturbation
# # Summary plot
# fig_perts, ax_perts = fp.get_figax(aspect=5)
# fig_summ, ax_summ[0] = format_plot(nstate)
# # BC = [vertical_shift, amplitude, frequency]
# BC1 = [2000, -200, 1]
# BC2 = [2000, 200, 1]
# BC3 = [2000, -100, 1]
# BC4 = [2000, -200, 2]
# BC5 = [2000, -200, 3]
# BCs = [BC1, BC2, BC3, BC4, BC5]
# for i, BC_l in enumerate(BCs):
#     BC = BC_l[0] + BC_l[1]*np.sin(BC_l[2]*2*np.pi/t.max()*t)

#     # Solve inversion
#     K = build_jacobian(x_a, y_init, BC, t, U, L, obs_t,
#                        optimize_BC)
#     y_a = forward_model_lw(x_a, y_init, BC, t, U, L, obs_t)
#     c = y_a.flatten() - K @ np.ones(K.shape[1])
#     x_hat, s_hat, a = solve_inversion(x_a, s_a,
#                                       y.flatten(), y_a.flatten(), s_o, K,
#                                       optimize_BC, verbose=True)

#     # # Plots
#     # Perturbation plot
#     ax_perts.plot(t/3600, BC, c=fp.color(i*2, lut=len(BCs)*2), lw=2)


#     # # Inversion plot
#     # fig, ax = plot_inversion(x_a, x_hat, x_true, x_hat_true,
#     #                          optimize_BC=optimize_BC)
#     # ax = fp.add_title(ax,
#     #                   f'Oscillating Boundary Condition\n(BC = {int(BC_l[0]):d} + {int(BC_l[1]):d}sin({int(BC_l[2]):d}at) ppb)')

#     # fp.save_fig(fig, plot_dir, f'oscillating_BC{i}_n{nstate}_m{nobs}')

#     fig, ax = plot_obs(nstate, y, y_a, y_ss)
#     ax.scatter(xp, c, c='red', s=30)
#     ax.scatter(xp, (K @ np.ones(K.shape[1])), c='blue', s=30)
#     print(BC)
#     print(c)

#     ax = fp.add_title(ax,
#                       f'Oscillating Boundary Condition\n(BC = {int(BC_l[0]):d} + {int(BC_l[1]):d}sin({int(BC_l[2]):d}at) ppb)')
#     fp.save_fig(fig, plot_dir, f'oscillating_BC{i}_n{nstate}_m{nobs}_obs')

#     # # Summary plot
#     # ax_summ.plot(xp, np.abs(x_hat - x_hat_true)*x_a*3600*24,
#     #              c=fp.color(i*2, lut=len(BCs)*2), lw=2, ls='-',
#     #              label=f'Test {i+1}')

# ax_perts.set_xlim(t.min()/3600, t.max()/3600)
# ax_perts.set_ylim(1600, 2300)
# ax_perts = fp.add_title(ax_perts, 'Oscillating Boundary Condition Perturbations')
# ax_perts = fp.add_labels(ax_perts, 'Time (hr)', 'BC (ppb)')
# fp.save_fig(fig_perts, plot_dir, f'oscillating_BC_perts_summary')

# if optimize_BC:
#     txt = 'BC optimized'
# else:
#     txt = 'BC not optimized'
# ax_summ.text(0.98, 0.95, txt, ha='right', va='top',
#              fontsize=config.LABEL_FONTSIZE*config.SCALE,
#              transform=ax_summ.transAxes)
# ax_summ.set_ylim(0, 100)

# fp.add_title(ax_summ, 'Oscillating Boundary Condition Perturbations')
# fp.add_labels(ax_summ, 'State Vector Element', r'$\Delta$XCH4 (ppb)')
# fp.add_legend(ax_summ, bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=3)
# fp.save_fig(fig_summ, plot_dir, f'oscillating_BC_summary_n{nstate}_m{nobs}')
# plt.close()

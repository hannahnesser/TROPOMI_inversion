# ------------------------------------------------------------------------ ##
# Permian comparison
# ------------------------------------------------------------------------ ##
# f = fs[5]
# dofs = np.load(f'{data_dir}posterior/dofs2_{f}.npy')
# xhat = np.load(f'{data_dir}posterior/xhat2_{f}.npy')
# shat = np.load(f'{data_dir}shat_permian.npy') # rg4rt_rf1.0_sax1.0_poi80
# dofs_l = 1 - shat

# Filter
# xhat[dofs < DOFS_filter] = 1
# dofs[dofs < DOFS_filter] = 0

# # Plot averaging kernel
# fig, ax = fp.get_figax(aspect=2)
# print(dofs_l.shape)
# for i in range(20):
#     shat_row = shat[i, :]
#     ax.plot(shat_row, c=fp.color(i, lut=20), lw=0.1)
# fp.save_fig(fig, plot_dir, f'fig_permian_dofs_rg4rt_rf1.0_sax1.0_poi80')

# Combine the Permian clusters with the NA clusters
permian = xr.open_dataset(f'{data_dir}clusters_permian.nc')['Clusters']
c = clusters.squeeze(drop=True).to_dataset()
c['Permian'] = permian

# Get the Permian basin indices (discard the buffer cells)
cell_idx, cell_cnt  = np.unique(c['Permian'], return_counts=True)
cell_idx = cell_idx[cell_cnt == 1]
cell_idx = cell_idx[~np.isnan(cell_idx)]
permian = permian.where(permian.isin(cell_idx), 0)

# # Subset over the Permian
# c = c.where(c['Permian'].isin(cell_idx))['Clusters']
# c = c.sel(lat=permian.lat, lon=permian.lon)

# # Flatten and create boolean
# permian_idx = (ip.clusters_2d_to_1d(permian, c) - 1).astype(int)
permian_idx = np.load(f'{data_dir}permian_idx.npy')
# print(c)
# c[c > 0] = 1

nstate_permian = len(permian_idx)

# for dofs_t in [0.01, 0.05, 0.1, 0.25]:
#     xhat_sub = xhat[dofs >= dofs_t]
#     ja = ((xhat_sub - 1)**2/4).sum()/(len(xhat_sub))
#     print(f'{dofs_t:<5}{xhat_sub.min():.2f}  {xhat_sub.max():.2f}  {ja:.2f}')

# Subset the posterior
# nscale = 1
xhat_permian = xhat[permian_idx, :]
# xhat_permian = ((xhat_permian - 1)*nscale + 1)
xa_abs_permian = xa_abs[permian_idx, :]
# xhat_abs_permian = xhat_permian*xa_abs_permian
xhat_abs_permian = xhat_abs[permian_idx, :]
# soil_permian = soil[permian_idx]
dofs_permian = dofs[permian_idx, :]
area_permian = area[permian_idx, :]
# a_permian = np.load(f'{data_dir}posterior/a2_rg4rt_edf_permian.npy')
# shat_permian = shat[:, permian_idx]

# fig, ax = fp.get_figax(aspect=1)
# cax = fp.add_cax(fig, ax)
# c = ax.matshow(shat_permian, cmap='RdBu_r', vmin=-0.001, vmax=0.001)
# cb = fig.colorbar(c, cax=cax)#, ticks=np.arange(0, 6, 1))
# cb = fp.format_cbar(cb, cbar_title=r'$\hat{S}$')
# fp.save_fig(fig, plot_dir, f'fig_permian_shat_rg4rt_rf1.0_sax1.0_poi80')

# # Subset the posterior errors (this needs to be done remotely)
# shat = np.load(f'/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results/iteration2/shat/shat2_rf1.0_sax1.0_poi80.npy')
# permian_idx = np.load(f'/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results/permian_idx.npy')
# xa_abs = xr.open_dataarray(f'/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/inversion_results/xa_abs.nc')
# xa_abs_permian = xa_abs[permian_idx]
# nstate_permian = len(xa_abs_permian)
# shat_permian = shat[permian_idx, permian_idx]
# shat_abs_permian = shat_permian*xa_abs_permian.values.reshape((-1, 1))*xa_abs_permian.values.reshape((1, -1))
# shat_abs_permian = xr.DataArray(shat_abs_permian, dims=('nstate1', 'nstate2'),
#                                 coords={'nstate1' : np.arange(1, nstate_permian+1),
#                                         'nstate2' : np.arange(1, nstate_permian+1)})
# shat_abs_permian.to_netcdf(f'/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/inversion_data/shat_abs_permian.nc')


# Save out
# permian.to_netcdf(f'{data_dir}clusters_permian.nc')

# xa_abs_permian = xr.DataArray(xa_abs_permian.reshape(-1,), 
#                               dims=('nstate'),
#                               coords={'nstate' : np.arange(1, nstate_permian+1)})
# # xa_abs_permian.to_netcdf(f'{data_dir}xa_abs_edf_permian.nc')

# xhat_abs_permian = xr.DataArray(xhat_abs_permian.reshape(-1,), 
#                                 dims=('nstate'),
#                                 coords={'nstate' : np.arange(1, nstate_permian+1)})
# # xhat_abs_permian.to_netcdf(f'{data_dir}xhat_abs_edf_permian.nc')

# Adjust units to Tg/yr
xa_abs_permian *= area_permian*1e-6 # Tg/yr
xhat_abs_permian *= area_permian*1e-6

# Calculate emissions
tot_prior_permian = xa_abs_permian.sum()
tot_post_permian = xhat_abs_permian.sum()
print(f'Minimum correction               : {xhat_permian.min()}')
print(f'Maximum correction               : {xhat_permian.max()}')
print(f'Median correction                : {np.median(xhat_permian)}')
print(f'Mean correction                  : {xhat_permian.mean()}')

print(f'Total prior emissions            : {tot_prior_permian}')
print(f'Total posterior emissions        : {tot_post_permian}')
print(f'Difference                       : {(tot_post_permian - tot_prior_permian)}')

# Adjust back to kg/km2/hr
xa_abs_permian = xa_abs_permian/area_permian/1e-9/(365*24)
xhat_abs_permian = xhat_abs_permian/area_permian/1e-9/(365*24)
# print(xa_abs_permian)
# print(xhat_permian)

fig, axis = fp.get_figax(rows=2, cols=2, maps=True,
                         lats=permian.lat, lons=permian.lon,
                         max_width=config.BASE_WIDTH*2.5,
                         max_height=config.BASE_HEIGHT*2.5)
plt.subplots_adjust(hspace=-0.05, wspace=0.75)

# Plot prior
fig_kwargs = {'figax' : [fig, axis[0, 0]]}
xhat_kwargs = {'cmap' : yor_trans, 'vmin' : 0, 'vmax' : 13,
               'default_value' : 0,
               'map_kwargs' : small_map_kwargs,
               'fig_kwargs' : fig_kwargs}
title = f'Prior emissions' # ({f}\%)'
fig, axis[0, 0], c = ip.plot_state(xa_abs_permian, permian, title=title, 
                                   cbar=False, **xhat_kwargs)
axis[0, 0].text(0.05, 0.05, f'{tot_prior_permian:.1f} Tg/yr',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*1,
                transform=axis[0, 0].transAxes)


# Plot posterior emissions
fig_kwargs = {'figax' : [fig, axis[0, 1]]}
xhat_cbar_kwargs = {'title' : r'Emissions\\(kg km$^{-2}$ h$^{-1}$)', 'x' : 4,
                    'cbar_pad_inches' : 0.1}
xhat_kwargs['fig_kwargs'] = fig_kwargs
xhat_kwargs['cbar_kwargs'] = xhat_cbar_kwargs
title = f'Posterior emissions' # ({f}\%)'
fig, axis[0, 0], c = ip.plot_state(xhat_abs_permian, permian, title=title, 
                                   **xhat_kwargs)
axis[0, 1].text(0.05, 0.05, f'{tot_post_permian:.1f} Tg/yr',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*1,
                transform=axis[0, 1].transAxes)

# # Plot posterior scaling factors
# sf_cmap_1 = plt.cm.Oranges(np.linspace(0, 1, 256))
# sf_cmap_2 = plt.cm.Purples(np.linspace(0.5, 0, 256))
# sf_cmap = np.vstack((sf_cmap_2, sf_cmap_1))
# sf_cmap = colors.LinearSegmentedColormap.from_list('sf_cmap', sf_cmap)
# div_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=3)

xhat_cbar_kwargs = {'title' : r'Scale factor', 'x' : 4, 
                    'ticks' : np.arange(0, 3.1, 0.5), 
                    'cbar_pad_inches' : 0.1}
fig_kwargs = {'figax' : [fig, axis[1, 0]]}
xhat_kwargs = {'cmap' : sf_cmap, 'norm' : div_norm,
               'default_value' : 1,
               'cbar_kwargs' : xhat_cbar_kwargs,
               'map_kwargs' : small_map_kwargs,
               'fig_kwargs' : fig_kwargs}
title = f'Posterior\nscale factors' # ({f}\%)'
fig, axis[1, 0], c = ip.plot_state(xhat_permian, permian, title=title,
                                   **xhat_kwargs)
axis[1, 0].text(0.05, 0.05,
                f'({(xhat_permian.min()):.1f}, {(xhat_permian.max()):.1f})',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*1,
                transform=axis[1, 0].transAxes)

# Plot DOFS
fig_kwargs = {'figax' : [fig, axis[1, 1]]}
avker_cbar_kwargs = {'title' : r'$\partial\hat{x}_i/\partial x_i$', 'x' : 4,
                     'cbar_pad_inches' : 0.1}
avker_kwargs = {'cmap' : plasma_trans, 'vmin' : 0, 'vmax' : 1,
                'cbar_kwargs' : avker_cbar_kwargs,
                'map_kwargs' : small_map_kwargs,
                'fig_kwargs' : fig_kwargs}
title = f'Averaging kernel\nsensitivities' # ({f}\%)'
fig, axis[1, 1], c = ip.plot_state(dofs_permian, permian, title=title,
                                 **avker_kwargs)
axis[1, 1].text(0.05, 0.05,
                f'DOFS = {dofs_permian.sum():.1f}',
                fontsize=config.LABEL_FONTSIZE*config.SCALE*1,
                transform=axis[1, 1].transAxes)


fp.save_fig(fig, plot_dir, f'fig_est2_xhat_permian_{f}')
plt.close()

fig, ax = fp.get_figax(aspect=1)
ax.hist(xa_abs_permian, bins=50, alpha=0.5)
ax.hist(xhat_abs_permian, bins=50, alpha=0.5)

# ax.scatter(xa_abs_permian, xhat_permian)
fp.save_fig(fig, plot_dir, f'fig_est2_xhat_permian_scatter_{f}')


# # Plot rows of the averaging kernel
# fig, ax = fp.get_figax(aspect=2)
# i = 0
# # for row in a_permian:
#     # ax.plot(row, c=fp.color(i, lut=nstate_permian), lw=0.1)
#     # i += 1
# ax.plot(a_permian[100, :], c=fp.color(100, lut=nstate_permian), lw=1)
# ax.set_xlim(1e4, 23691)
# fp.save_fig(fig, plot_dir, f'fig_est2_a_permian_{f}')

# print(i)


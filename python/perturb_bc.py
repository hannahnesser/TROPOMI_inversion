import xarray as xr
import numpy as np
import glob
import copy

# Define directories
code_dir = '/n/holyscratch01/jacob_lab/hnesser/TROPOMI_inversion/python/'
bc_dir = '/n/jacob_lab/Lab/seasasfs02/hnesser/TROPOMI_inversion/'

# Load custom packages
import sys
sys.path.append(code_dir)
# sys.path.append('.')
import inversion_settings as s
import invpy as ip
import format_plots as fp
import matplotlib.pyplot as plt

# Get boundary conditions
bcs = glob.glob(f'{bc_dir}boundary_conditions/GEOSChem.BoundaryConditions.2019*')
bcs.sort()
# bcs = ['../inversion_data/GEOSChem.BoundaryConditions.20191124_0000z.nc4']

# Get boundary condition edges
bcf = xr.open_dataset(bcs[0])['SpeciesBC_CH4']
lat_e = ((bcf.lat.values[:-1] + bcf.lat.values[1:])/2)[1:-1]
lat_e = np.hstack((lat_e.min() - 2, lat_e, lat_e.max() + 2))
lon_e = ((bcf.lon.values[:-1] + bcf.lon.values[1:])/2)[1:-1]
lon_e = np.hstack((lon_e.min() - 2, lon_e, lon_e.max() + 2))

# Get domain edges (smaller, larger)
nn = (s.lat_max, s.lat_max + s.buffers[0]*s.lat_delta)
ee = (s.lon_max, s.lon_max + s.buffers[2]*s.lon_delta)
ss = (s.lat_min - s.buffers[1]*s.lat_delta, s.lat_min)
ww = (s.lon_min - s.buffers[3]*s.lon_delta, s.lon_min)

# Find boundary condition edges containing domain edges (smaller, larger)
nbc = (lat_e[lat_e < nn[0]][-1], lat_e[lat_e > nn[1]][0])
ebc = (lon_e[lon_e < ee[0]][-1], lon_e[lon_e > ee[1]][0])
sbc = (lat_e[lat_e < ss[0]][-1], lat_e[lat_e > ss[1]][0])
wbc = (lon_e[lon_e < ww[0]][-1], lon_e[lon_e > ww[1]][0])

print('-'*100)
for bc in bcs:
    fname = bc.split('/')[-1]
    print(fname)
    bcf = xr.open_dataset(bc)

    # Northern boundary
    bcf_n = copy.deepcopy(bcf)
    bcf_n['SpeciesBC_CH4'].loc[{'lat' : slice(*nbc),
                                'lon' : slice(wbc[0], ebc[1])}] += 10*1e-9
    # bcf_n['SpeciesBC_CH4'].loc[{'lat' : slice(*nbc),
    #                             'lon' : slice(wbc[0], ebc[1])}] *= 1.5
    bcf_n.to_netcdf(f'{bc_dir}boundary_conditions_N/{fname}')

    # Eastern boundary
    bcf_e = copy.deepcopy(bcf)
    bcf_e['SpeciesBC_CH4'].loc[{'lat' : slice(sbc[1], nbc[0]),
                                'lon' : slice(*ebc)}] += 10*1e-9
    # bcf_e['SpeciesBC_CH4'].loc[{'lat' : slice(sbc[1], nbc[0]),
    #                             'lon' : slice(*ebc)}] *= 1.5
    bcf_e.to_netcdf(f'{bc_dir}boundary_conditions_E/{fname}')

    # Southern boundary
    bcf_s = copy.deepcopy(bcf)
    bcf_s['SpeciesBC_CH4'].loc[{'lat' : slice(*sbc),
                                'lon' : slice(wbc[0], ebc[1])}] += 10*1e-9
    # bcf_s['SpeciesBC_CH4'].loc[{'lat' : slice(*sbc),
    #                             'lon' : slice(wbc[0], ebc[1])}] *= 1.5
    bcf_s.to_netcdf(f'{bc_dir}boundary_conditions_S/{fname}')

    # Western boundary
    bcf_w = copy.deepcopy(bcf)
    bcf_w['SpeciesBC_CH4'].loc[{'lat' : slice(sbc[1], nbc[0]),
                                'lon' : slice(*wbc)}] += 10*1e-9
    # bcf_w['SpeciesBC_CH4'].loc[{'lat' : slice(sbc[1], nbc[0]),
    #                             'lon' : slice(*wbc)}] *= 1.5
    bcf_w.to_netcdf(f'{bc_dir}boundary_conditions_W/{fname}')


# for f in ['N', 'E', 'S', 'W']:
#     f = xr.open_dataset(f'../inversion_data/GEOSChem.BoundaryConditions.20190102_0000z_{f}.nc4')
#     f = f.sel(lev=f.lev[0], time=f.time[0])['SpeciesBC_CH4']
#     fig, ax = fp.get_figax(maps=True, lats=f.lat, lons=f.lon)
#     f.plot(ax=ax)
#     ax = fp.format_map(ax, f.lat, f.lon)
#     ax.axvline(s.lon_min, color='black')
#     ax.axvline(s.lon_max, color='black')
#     ax.axhline(s.lat_min, color='black')
#     ax.axhline(s.lat_max, color='black')
#     ax.set_xlim(ww[0], ee[1])
#     ax.set_ylim(ss[0], nn[1])
#     plt.show()

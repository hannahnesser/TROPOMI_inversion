import xarray as xr

cat = '1B2ci1'
old_file = './old/can_emis_%s_2018.nc' % cat
new_file = './new/can_emis_%s_2018.nc' % cat

old = xr.open_dataset(old_file)
new = xr.open_dataset(new_file)

print(old)
print(new)

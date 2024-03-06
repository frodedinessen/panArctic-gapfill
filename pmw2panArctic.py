#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 08:47:33 2023

@author: froded
"""

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import pmw2asip as pa

NoData = -1

# OSI SAF L3 SIC (polar arcic grid)
#amsr_fn = 'https://thredds.met.no/thredds/dodsC/osisaf/met.no/ice/amsr2_conc/2023/12/ice_conc_nh_polstere-100_amsr2_202312061200.nc'
#amsr_fn = '/home/froded/data/panArctic/istjnh3km_202310302000-202310310800.nc'
#amsr_fn = 'https://thredds.met.no/thredds/dodsC/myocean/siw-tac/siw-metno-glo-osisaf/conc/2020/05/ice_conc_nh_polstere-100_multi_202005251200.nc'

#amsr_fn = 'https://thredds.met.no/thredds/dodsC/cosi/AMSR2_SIC/v1_April2023/2022/05/sic_cosi-5km_202205010000-202205020000.nc'
#amsr_fn = 'https://thredds.met.no/thredds/dodsC/osisaf/met.no/ice/amsr2_conc/2023/05/ice_conc_nh_polstere-100_amsr2_202305011200.nc'
amsr_fn = 'https://thredds.met.no/thredds/dodsC/osisaf/met.no/ice/amsr2_conc/2022/05/ice_conc_nh_polstere-100_amsr2_202205011200.nc'

# DMI AISP L3 file
#asip_fn = '/home/froded/data/panArctic/dmi_asip_panarctic_seaice_mosaic_20200525-20200531.nc'
#asip_fn = '/home/froded/data/panArctic/dmi_asip_panarctic_seaice_mosaic_20230501.nc

asip_fn = '/home/froded/data/panArctic/dmi_asip_panarctic_seaice_mosaic_20220501.nc'
#asip_fn = '/home/froded/data/panArctic/dmi_asip_panarctic_seaice_mosaic_20230501.nc'


# Map the OSISAF L3 AMSR2 SIC data into projection and area and resolution of the ASIP product
# The OSISAF data are first resampled into the resolution trg_res using pyresampled.
# Then a linear upsampling is used to match the AISP resolution. 
trg_res = 3000
pmwsic = 'ice_conc'
#pmwstd = 'total_standard_uncertainty' # COSI
pmwstd = 'total_uncertainty' # OSI-408

asipsic = 'ice_concentration'
asipstd = 'uncertainty'
asipflag = 'status_flag'


#varname = 'ice_conc_n90'
trg_dat = pa.pmw2asip(amsr_fn,asip_fn, trg_res, pmwsic)

# Set nodato
#trg_dat[trg_dat>101]=NoData

# Find idx where AMSR2 has ice
#idx_iceAMSR2 = np.where((trg_dat>=0) & (trg_dat<=101))

# Read ASIP data
asip_ds = xr.open_dataset(asip_fn)
asip = asip_ds[asipsic][0].data

# Make a mask of pixels use from AMSR2 data AMSR2 -> ASIP
asipLand = 1
if asipLand:
    # Create mask where ASIP have NaN and AMSR2 has SIC
    idx_p2a = ((np.isnan(asip)) & (trg_dat>=0) & (trg_dat<=101))

else:
    # Create mask where ASIP have NaN and AMSR2 has SIC,Land and NoData
    asip[asip>101]=np.nan
    idx_p2a = ((np.isnan(asip)) & (trg_dat>=0))

# Use mask to copy data from AMSR2 -> ASIP
asip[idx_p2a] = trg_dat[idx_p2a]

if not asipLand:
    # Set nodato and PMW land
    asip[trg_dat>101]=np.nan

# Fill the pole hole
idx_ph, asip = pa.fill_pole_hole(asip)


# Find index of NaN and asip land after merging ASIP and AMSR2
# If landmask from PMW 
idx_nanAsip =((np.isnan(asip)) | (asip==120))

# Set NaN and Land to _FillValue = -1
asip[idx_nanAsip] = NoData


# Add a time layer 
asip = asip[None,:,:]
 
# Generate a netCDF tamplate for writing variables
ds_temp = pa.to_cf_template(asip_ds, skip_lonlat=False)

# Put variables into the template
da_var2 = xr.DataArray(asip, coords=ds_temp['template'].coords, dims=('time', *ds_temp['template'].dims),
                       attrs=ds_temp['template'].attrs, name=asipsic)
da_var2.attrs['long_name'] = asip_ds[asipsic].long_name
da_var2.attrs['units'] = asip_ds[asipsic].units
da_var2.attrs['_FillValue'] = NoData
ds_temp = ds_temp.merge(da_var2)

# Variable name PMW data
trg_dat = pa.pmw2asip(amsr_fn,asip_fn, trg_res, pmwstd)
trg_dat[trg_dat>101]=NoData

asip_std = asip_ds[asipstd][0].data 
asip_std[idx_p2a] = trg_dat[idx_p2a]
asip_std[idx_ph] = np.nan

# Fill the pole hole
idx_ph, asip_std = pa.fill_pole_hole(asip_std)
asip_std[idx_nanAsip] = NoData
asip_std = asip_std[None,:,:]


da_var_std = xr.DataArray(asip_std, coords=ds_temp['template'].coords, dims=('time', *ds_temp['template'].dims),
                       attrs=ds_temp['template'].attrs, name=asipstd)
da_var_std.attrs['long_name'] = asip_ds[asipstd].long_name
da_var_std.attrs['units'] = asip_ds[asipstd].units
da_var_std.attrs['_FillValue'] = NoData
ds_temp = ds_temp.merge(da_var_std)


status_flag = np.ones(asip_std[0].shape, dtype=np.byte)
status_flag[idx_p2a] = 2
status_flag[idx_nanAsip] = NoData
status_flag[idx_ph] = 3
status_flag = status_flag[None,:,:]

da_var_flag = xr.DataArray(status_flag, coords=ds_temp['template'].coords, dims=('time', *ds_temp['template'].dims),
                       attrs=ds_temp['template'].attrs, name=asipflag)
da_var_flag.attrs['long_name'] = 'status flag for sea ice concentration retrieval'
da_var_flag.attrs['units'] = 1
da_var_flag.attrs['_FillValue'] = NoData
da_var_flag.attrs['standard_name'] = 'sea_ice_area_fraction status_flag'
da_var_flag.attrs['flag_values'] = [0, 1, 2, 3]
da_var_flag.attrs['status_flag'] = "flag_descriptions = \n,  -1 -> NoData and Land\n,  1 -> sea ice concentration from ASIP (SAR+AMSR2)\n,  \
    2 -> Sea ice concentration from AMSR2 data 3 -> interpolated\n"
ds_temp = ds_temp.merge(da_var_flag)


ds_temp = ds_temp.drop('template')

#o_nc = '/home/froded/data/panArctic/dmi_asip_panarctic_seaice_mosaic_20220501_osi_408_202305011200_landASIP.nc'
o_nc = '/home/froded/data/panArctic/dmi_asip_panarctic_seaice_mosaic_20220501_osi_408_12000_40.nc'
ds_temp.to_netcdf(o_nc, format='NETCDF4_CLASSIC')

##############################################################




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:59:02 2024

@author: froded
"""

import xarray as xr
import pyresample as pr
from pyresample.geometry import AreaDefinition
import numpy as np
from scipy import interpolate
from datetime import datetime, timedelta


def fill_pole_hole(arr):
    # Select an area covering the pole hole
    xmin = 3650
    xmax  = 3850
    ymin = 5970
    ymax = 6170
     
    subarr = arr[ymin:ymax,xmin:xmax]

    # Get indexes for the selected area
    (yt,xt) = subarr.shape
    x = np.array(range(xt))
    y = np.array(range(yt))
    
    # Generate a masked array of nan
    subarr = np.ma.masked_invalid(subarr)
    
    # Get index for polar hole to interpolate
    idx_ph = np.zeros(arr.shape,dtype=np.byte)
    idx_ph[ymin:ymax,xmin:xmax] = subarr.mask
    idx_ph = np.where(idx_ph)
    
    # Select valid data pixels
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~subarr.mask]
    y1 = yy[~subarr.mask]
                                         
    newarr = subarr[~subarr.mask]
    
    newarr = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy), method='cubic') 
    
    # Fill interpolated data into the input array 
    arr[ymin:ymax,xmin:xmax] = newarr
    
    return idx_ph, arr
    

def upsampling(trg_dat, asip_ds):
    (yt,xt) = trg_dat.shape
    (yd,xd) = asip_ds['ice_concentration'][0].shape

    x = np.array(range(xt))
    y = np.array(range(yt))
    f = interpolate.interp2d(x, y, trg_dat.data, kind='linear')
    #m = interpolate.interp2d(x, y, trg_dat.mask, kind='linear')
    xnew = np.linspace(0, xt, xd)
    ynew = np.linspace(0, yt, yd)

    trg_ups = f(xnew,ynew)

    return trg_ups

def resampling(amsr_adef, trg_adef, dataArr):
    # Parameters for the remapping

    # Radius of Influence (meters): the maximum distance to look for "source" points (neighbours)
    #   that can contribute to the remapped value in one "target" grid cell.
    #roi = 6000 # COSI
    roi = 12000
    # Number of Neighbours (defaults to 8): the maximum number of "source" points that are returned by the
    #   the radius_of_influence search. Selecting  a large ROI without changing this does not help.
    #nbgh = 25
    nbgh = 40
    
    # Sigma (meters): once we have our neighbours selected (two parameters above), how to weight
    #    their contribution as a function of their distance to the "target" grid cell (center).
    sigma = roi/2.5
    
    # fill_value: what value to put in the output (remapped) field in cells where no neighbours existed.
    #   fill_value = None means a Masked Array is returned
    fill_value = None
    
    # These lat/lons are computed by pyresample from the area definition
    prlons, prlats = amsr_adef.get_lonlats()

    # the field to be remapped
    src_dat = dataArr.to_masked_array()

    # the geometry (lat/lon) of the cells to be remapped.
    #    here we remove the missing values, which turns the 2D lat/lon into 1D arrays 
    src_lon_dat = prlons[~src_dat.mask]
    src_lat_dat = prlats[~src_dat.mask]
    src_geom_dat = pr.SwathDefinition(src_lon_dat, src_lat_dat)

    # Prepare the data in the same 1D shape as the src lat/lon
    src_dat_1d = src_dat.compressed()

    # do the remapping
    trg_dat = pr.kd_tree.resample_gauss(src_geom_dat, src_dat_1d, trg_adef,
                                        radius_of_influence=roi, sigmas=sigma, neighbours=nbgh,
                                        fill_value=fill_value, reduce_data=False)

    return trg_dat
    

def pmw2asip(amsr_fn,asip_fn, trg_res, varname):
    
    # use pyresample to get an AreaDefinition object
    amsr_ds = xr.open_dataset(amsr_fn)
    amsr_adef, amsr_cfinfo = pr.utils.load_cf_area(amsr_ds)
    
    # Define target output based on DMI L3 product.
    # Mapping AMSR2 into 5000m resolution (DMI is 1000m)
    asip_ds = xr.open_dataset(asip_fn)
    # asip_adef, asip_cfinfo = pr.utils.load_cf_area(asip_ds)
    
    # asip_ext = asip_adef.area_extent
    
    # # Update resolution based on trg_res
    # asip_adef.width = round((asip_ext[2]-asip_ext[0])/trg_res)
    # asip_adef.height = round((asip_ext[3]-asip_ext[1])/trg_res)
    
    # Define an AreaDefinitions for trg_dat based on asip_ds projection and extent. 
    area_id = 'crs'
    description = 'crs'
    proj_id = 'stere_nh'
    projection = asip_ds['crs'].attrs['proj4_string']
    
    # Find dimentions of AMSR2 resampled to 5000 m
    (xmin,xmax) = asip_ds['x'][0].data,asip_ds['x'][-1].data
    (ymin,ymax) = asip_ds['y'][-1].data,asip_ds['y'][0].data
    
    width = round((xmax-xmin)/trg_res)
    hight = round((ymax-ymin)/trg_res)
    print(width,hight)

    area_extent = (xmin.item(),  ymin.item(), xmax.item(),  ymax.item())
    trg_adef = AreaDefinition(area_id, description, proj_id, projection, width, hight, area_extent)
    print(trg_adef)
    print("Grid spacing:", trg_adef.projection_x_coords[1]-trg_adef.projection_x_coords[0], '[m]')
    
    trg_dat = resampling(amsr_adef, trg_adef, amsr_ds[varname][0])
    
    
    trg_dat = upsampling(trg_dat, asip_ds)
    
    amsr_ds.close()
    asip_ds.close()
    
    return trg_dat
    
    
def to_cf_template(asip_ds, skip_lonlat=True):
    """Return a template xarray Dataset holding the structure of a netCDF/CF file for this grid."""
    
    # Area definition from asip dataset
    area_def, asip_cfinfo = pr.utils.load_cf_area(asip_ds)
    
    # prepare the crs object with pyproj.to_cf()
    crs_cf = area_def.crs.to_cf()
    type_of_grid_mapping = crs_cf['grid_mapping_name']
    
    # prepare the x and y axis (1D)
    xy = dict()
    xy_dims = ('x', 'y')
    
    for axis in xy_dims:
        
        # access the valid standard_names (also handle the 'default')
        try:
            valid_coord_standard_names = pr.utils.cf._valid_cf_coordinate_standardnames[type_of_grid_mapping][axis]
        except KeyError:
            valid_coord_standard_names = pr.utils.cf._valid_cf_coordinate_standardnames['default'][axis]
        
        xy[axis] = dict()
        # CF wants the center of the grid cells
        if axis == 'x':
            xy[axis]['_coords'] = area_def.projection_x_coords
        else:
            xy[axis]['_coords'] = area_def.projection_y_coords     
        # each axis requires a valid name, which depends on the type of projection
        xy[axis]['standard_name'] = valid_coord_standard_names[0]
        # CF recommendation to have axis= attribute
        xy[axis]['axis'] = axis.upper()
        # units
        xy[axis]['units'] = 'm'
    
    # latitude and longitude (2D)
    lons, lats = area_def.get_lonlats()
    
    # determine the order of the x, y dimensions that match the shape of lat/lon arrays
    if lons.shape == (len(xy['x']['_coords']),len(xy['y']['_coords'])):
        xy_dims = ('x', 'y')
    elif lons.shape == (len(xy['y']['_coords']),len(xy['x']['_coords'])):
        xy_dims = ('y', 'x')
    else:
        raise ValueError("Incompatible shape for lon/lat {}, x {}, and y {}.".format(lons.shape,
                                                                                    len(xy['x']['_coords']),
                                                                                    len(xy['y']['_coords'])))
    
    # define a Dataset as a template.
    # The strategy is to 
    # create the Dataset with an (empty) variable, and the user can later 
    # add his own variables, and finally drop our 'template' variable before
    # writing to file.
    varn = 'template'
    shape = lons.shape
    
    # time variable 
    reference_date = datetime(1970, 1, 1, 0, 0, 0)
    time_diff = get_days_since_ref(asip_ds,reference_date)
    
    da_empty_data = np.ones_like(lons) * np.nan
    da_dims = list(xy_dims)
    
    da_coords = {'x':('x',xy['x']['_coords']), 'y':('y',xy['y']['_coords']),}
    da_coords['time'] = ('time'), np.array([time_diff], dtype=np.double)
    
    if not skip_lonlat:
        da_coords['lon']=(da_dims, lons)
        da_coords['lat']=(da_dims, lats)
    ds = xr.Dataset(data_vars={varn:(da_dims, da_empty_data),
                               'crs':([], 0)},
                    coords=da_coords,)
        
    # add CF attributes and encodings to the xarray template
    
    # time dims    
    ds['time'].attrs={'long_name': 'reference time of product', 'standard_name': 'time', 
                      'units': 'Days since 1970-01-01 00:00:00', 'calendar': 'standard'}
    ds['time'].encoding = {'_FillValue':None}
    
    # x and y dims
    for axis in xy_dims:
        for attr in xy[axis].keys():
            if attr.startswith('_'): continue
            ds[axis].attrs[attr] = xy[axis][attr]
        ds[axis].encoding = {'_FillValue':None}
    
    # crs object
    ds['crs'].attrs = crs_cf
    ds['crs'].encoding = {'dtype':'int32'}
    
    # latitude and longitude
    if not skip_lonlat:
        ds['lon'].attrs={'long_name': 'longitude coordinate', 'standard_name': 'longitude', 'units': 'degrees_east'}
        ds['lat'].attrs={'long_name': 'latitude coordinate', 'standard_name': 'latitude', 'units': 'degrees_north'}
        ds['lon'].encoding = {'_FillValue':None}
        ds['lat'].encoding = {'_FillValue':None}
        
    # the empty variable itself
    ds[varn].attrs['grid_mapping'] = 'crs'
    
    return ds


def get_days_since_ref(ds,reference_date):
    
    dt64 = ds['time'].data[0]
    if dt64.dtype =='<M8[ns]':
        ns = 1e-9
        dt = datetime.utcfromtimestamp(dt64.astype(int) * 1e-9)
    else:
        raise ValueError("Unable to read time from AMSR2 data. Expect AMSR time in nano seconds.") 
    
    # Calculate the time difference
    time_difference = dt - reference_date
    
    # Extract the time difference in days since reference 
    time_difference = time_difference.total_seconds()/(60*60*24)
    
    return time_difference
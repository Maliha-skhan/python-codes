# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:17:43 2020

@author: Maliha
"""
# import pckg
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import warnings
#from netCDF4 import Dataset
#from cartopy import config
import cartopy.crs as ccrs
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import datetime
#import time
#from scipy.signal import detrend
#%matplotlib inline
np.set_printoptions(precision=3, linewidth=80, edgeitems=1) # make numpy less verbose
xr.set_options(display_width=70)
warnings.simplefilter('ignore') # filter some warning messages


ncname1='CERES_EBAF-SWDO_Timemean.nc'
ncname2='CERES_EBAF-SWUP_Timemean.nc'
ncname3='CERES_EBAF-OLR_Timemean.nc'

filename = ncname1
#dataset=  Dataset(filename,'r')
ds=xr.open_dataset(filename)

long = ds.variables['lon'][:]
lati = ds.variables['lat'][:]
time = ds.variables['time'][:]
time=time.data

time_bnds = ds.variables['time_bnds'][:]
time_bnds=time_bnds.data

solar_mon = ds.variables['solar_mon'][:]
solar_mon=solar_mon.data
solar_mon=ds.solar_mon

plt.figure(tight_layout=True)
ax = plt.axes(projection=ccrs.PlateCarree())

solar_mon.plot(cmap='seismic') # bedst 
ax.set_title('Incoming solar flux')
#tccp.plot(cmap='tab20c') # flot men har for mange farver
#tccp.plot(cmap='Pastel2') # også ret pæn
ax.coastlines()



filename = ncname2
#dataset=  Dataset(filename,'r')
ds=xr.open_dataset(filename)
print(ds)
#long = ds.variables['lon'][:]
#lati = ds.variables['lat'][:]
#time = ds.variables['time'][:]
#time=time.data
#
#time_bnds = ds.variables['time_bnds'][:]
#time_bnds=time_bnds.data
#
toa_sw_all_mon = ds.variables['toa_sw_all_mon'][:]
toa_sw_all_mon=toa_sw_all_mon.data
toa_sw_all_mon=ds.toa_sw_all_mon

plt.figure(tight_layout=True)
ax = plt.axes(projection=ccrs.PlateCarree())

toa_sw_all_mon.plot(cmap='seismic') # bedst 
ax.set_title('(TOA) Shortwave flux')
#tccp.plot(cmap='tab20c') # flot men har for mange farver
#tccp.plot(cmap='Pastel2') # også ret pæn
ax.coastlines()


filename = ncname3
#dataset=  Dataset(filename,'r')
ds=xr.open_dataset(filename)
print(ds)

toa_lw_all_mon = ds.variables['toa_lw_all_mon'][:]
toa_lw_all_mon=toa_lw_all_mon.data
toa_lw_all_mon=ds.toa_lw_all_mon

plt.figure(tight_layout=True)
ax = plt.axes(projection=ccrs.PlateCarree())

toa_lw_all_mon.plot(cmap='seismic') # bedst 
ax.set_title('(TOA) Longwave flux')
#tccp.plot(cmap='tab20c') # flot men har for mange farver
#tccp.plot(cmap='Pastel2') # også ret pæn
ax.coastlines()



net_SW = solar_mon-toa_sw_all_mon

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
net_SW.plot(cmap=plt.cm.coolwarm)
plt.title('Net shortwave long time average')


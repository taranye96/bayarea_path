#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:23:42 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import rasterio

# Read in gmprocess records file
records_df = pd.read_csv('/Users/tnye/bayarea_path/data/gmprocess/waveforms/data/bayarea_default_metrics_rotd(percentile=50.0).csv')

# Get list of unique stations
stns = np.unique(records_df['StationCode'])[1:] #for some reason the header was showing up in this array

# Read in Vs30 raster
vs30_raster = rasterio.open('/Users/tnye/bayarea_path/files/vs30/California_vs30_Wills15_hybrid.tif')

# Read all the data from the first band
vs30_raster_data = vs30_raster.read()[0]

vs30_list = np.array([])
stlon_list = np.array([])
stlat_list = np.array([])

# Loop over stations
for stn in stns:
    
    # Get coordinates of the station
    stn_idx = np.where(records_df['StationCode']==stn)[0][0]
    stlon = records_df['StationLongitude'].values[stn_idx]
    stlat = records_df['StationLatitude'].values[stn_idx]
    
    # Extract Vs30
    vs30_idx = vs30_raster.index(stlon, stlat, precision=1E-6)   
    vs30 = vs30_raster_data[vs30_idx]
    vs30_list = np.append(vs30_list,vs30)

    stlon_list = np.append(stlon_list,stlon)
    stlat_list = np.append(stlat_list,stlat)

data = {'Station':stns, 'Stlon':stlon_list, 'Stlat':stlat_list, 'Vs30(m/s)': vs30_list}
vs30_df = pd.DataFrame(data)
vs30_df.to_csv('/Users/tnye/bayarea_path/files/vs30/bayarea_station_vs30.csv',index=False)

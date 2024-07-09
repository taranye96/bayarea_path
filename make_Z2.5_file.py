#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:03:51 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd

# Read in gmprocess records file
records_df = pd.read_csv('/Users/tnye/bayarea_path/data/gmprocess/waveforms/data/bayarea_default_metrics_rotd(percentile=50.0).csv')

# Get list of unique stations
stns = np.unique(records_df['StationCode'])[1:] #for some reason the header was showing up in this array
lons = []
lats = []
for stn in stns:
    lons.append(records_df['StationLongitude'].iloc[np.where(records_df['StationCode'] == stn)[0][0]])
    lats.append(records_df['StationLatitude'].iloc[np.where(records_df['StationCode'] == stn)[0][0]])
    
stn_list = []
z1_list = []
lon_list = []
lat_list = []

# Loop over stations
for i, stn in enumerate(stns):
    
    try:
        # Initialize input file
        coords = np.genfromtxt(f'/Users/tnye/bayarea_path/files/velmod/station_Vs/{stn}_Vs.out')[0,:2]
        data = np.genfromtxt(f'/Users/tnye/bayarea_path/files/velmod/station_Vs/{stn}_Vs.out')[:,2:]
        idx = np.where(data[:,1]>=2500)[0][0]
        z1 = -1*data[idx,0]
        stn_list.append(stn)
        z1_list.append(z1)
        lon_list.append(lons[i])
        lat_list.append(lats[i])
    
    except:
        print(f'{stn} {coords}: {np.max(data[:,1])}')
        continue

data = {'Station':stn_list, 'Longitude':lon_list, 'Latitude':lat_list, 'Z1.0(m)':z1_list}
df = pd.DataFrame(data)
df.to_csv('/Users/tnye/bayarea_path/files/site_info/station_z2.5.csv',index=False)

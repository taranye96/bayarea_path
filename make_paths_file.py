#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:41:18 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/tnye/bayarea_path/data/gmprocess/waveforms/data/bayarea_default_metrics_rotd(percentile=50.0).csv')

min_stns = 100
min_evs = 100

#%% Cull records to sations that have recorded at least 8 events and events
  # That have been recorded on at least 5 stations
  
stn_idx = np.array([])
ev_idx = np.array([])

events = np.unique(df['EarthquakeId'])
stns = np.unique(df['StationCode'])

for stn in stns:
    idx = np.where(df['StationCode'] == stn)[0]
    if len(idx) >= min_evs:
        stn_idx = np.append(stn_idx,idx)
    
for ev in events:
    idx = np.where(df['EarthquakeId'] == ev)[0]
    if len(idx) >= min_stns:
        ev_idx = np.append(ev_idx,idx)

useable_idx = np.intersect1d(stn_idx,ev_idx).astype(int)


#%%

hyplon = df['EarthquakeLongitude'].values[useable_idx]
hyplat = df['EarthquakeLatitude'].values[useable_idx]
stlon = df['StationLongitude'].values[useable_idx]
stlat = df['StationLatitude'].values[useable_idx]

# Find the indicies of the events within Bay Area geographic range
bay_ind = np.where((stlat > 36) & (stlat < 39) &
                    (stlon > -123.5 ) & (stlon < -121))

hyplon = hyplon[bay_ind]
hyplat = hyplat[bay_ind]
stlon = stlon[bay_ind]
stlat = stlat[bay_ind]

outfile = f'/Users/tnye/bayarea_path/files/paths/gmprocess_2000-2024_paths_{min_evs}evs-{min_stns}stns.txt'


f = open(outfile, 'w')

for i in range(len(hyplon)):
    f.write(">-Z1.515476448\n")
    f.write(f"{hyplon[i]}\t{hyplat[i]}\n")
    f.write(f"{stlon[i]}\t{stlat[i]}\n")

f.close()


#%%

# outfile = '/Users/tnye/bayarea_path/files/gmprocess_2000-2024_paths_pygmt.txt'

# f = open(outfile, 'w')

# for i in range(len(df)):
#     f.write(f"{hyplon[i]}\t{hyplat[i]}\n")
#     f.write(f"{stlon[i]}\t{stlat[i]}\n")

# f.close()
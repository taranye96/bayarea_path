#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:41:18 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/tnye/bayarea_path/data/gmprocess/waveforms_test/data/bayarea_default_metrics_geometricmean().csv')
hyplon = df['EarthquakeLongitude'].values
hyplat = df['EarthquakeLatitude'].values
stlon = df['StationLongitude'].values
stlat = df['StationLatitude'].values

# Find the indicies of the events within Bay Area geographic range
bay_ind = np.where((stlat > 36) & (stlat < 39) &
                    (stlon > -123.5 ) & (stlon < -121))

hyplon = hyplon[bay_ind]
hyplat = hyplat[bay_ind]
stlon = stlon[bay_ind]
stlat = stlat[bay_ind]

outfile = '/Users/tnye/bayarea_path/files/gmprocess_2000-2024_paths.txt'


f = open(outfile, 'w')

for i in range(len(hyplon)):
    f.write(">-Z1.515476448\n")
    f.write(f"{hyplon[i]}\t{hyplat[i]}\n")
    f.write(f"{stlon[i]}\t{stlat[i]}\n")

f.close()


#%%

outfile = '/Users/tnye/bayarea_path/files/gmprocess_2000-2024_paths_pygmt.txt'

f = open(outfile, 'w')

for i in range(len(df)):
    f.write(f"{hyplon[i]}\t{hyplat[i]}\n")
    f.write(f"{stlon[i]}\t{stlat[i]}\n")

f.close()
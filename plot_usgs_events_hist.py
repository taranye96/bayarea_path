#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:49:27 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from os import path, makedirs
import matplotlib.pyplot as plt

#%%

# Read in NGA-West2 finite fault file
event_catalog = pd.read_csv('/Users/tnye/bayarea_path/data/gmprocess/waveforms_test/data/bayarea_default_events.csv')

# Get cooordinates of events
hyplat_list = event_catalog['latitude'].values
hyplon_list = event_catalog['longitude'].values

# Find the indicies of the events within Bay Area geographic range
bay_ev_ind = np.where((hyplat_list > 36) & (hyplat_list < 39) &
                    (hyplon_list > -123.5 ) & (hyplon_list < -121))

# Get the magnitudes of the events in the Bay Area
mag_list = event_catalog['magnitude'].values[bay_ev_ind]

# Get depths of the events
depth_list = event_catalog['depth'].values[bay_ev_ind]

#%%

# Histogram of magnitudes
plt.figure()
plt.hist(mag_list,edgecolor='k')
plt.xlabel('Magnitude')
plt.ylabel('Counts')
plt.savefig('/Users/tnye/bayarea_path/plots/usgs_events_maghist.png',dpi=300)

# Histogram of depths
plt.figure()
plt.hist(depth_list,edgecolor='k')
plt.xlabel('Depth')
plt.ylabel('Counts')
plt.savefig('/Users/tnye/bayarea_path/plots/usgs_events_depth-hist.png',dpi=300)
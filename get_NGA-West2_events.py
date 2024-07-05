#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:05:53 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from os import path, makedirs
import matplotlib.pyplot as plt

#%%

# Read in NGA-West2 finite fault file
finitefault_df = pd.read_csv('/Users/tnye/bayarea_path/files/NGA_West2_supporting_data_for_flatfile/NGA_West2_Finite_Fault_Info_050142013.csv')

# Get cooordinates of events
hyplat_list = finitefault_df['Hypocenter Latitude (deg)'].values
hyplon_list = finitefault_df['Hypocenter Longitude (deg)'].values

# Find the indicies of the events within Bay Area geographic range
bay_ev_ind = np.where((hyplat_list > 36) & (hyplat_list < 39) &
                   (hyplon_list > -123.5 ) & (hyplon_list < -121))

# Get the magnitudes of the events in the Bay Area
mag_list = finitefault_df['Moment Magnitude'].values[bay_ev_ind]

# Get the names of the events in the Bay Area
event_list = finitefault_df['Earthquake Name'].values[bay_ev_ind]

# Make folders for the events to store the waveforms
for event in event_list:
    if not path.exists(f'/Users/tnye/bayarea_path/data/NGA-West2/{event}'):
        makedirs(f'/Users/tnye/bayarea_path/data/NGA-West2/{event}')
   
    
#%%

# Read in NGA-West2 station file
station_df = pd.read_csv('/Users/tnye/bayarea_path/files/NGA_West2_supporting_data_for_flatfile/NGA_West2_SiteDatabase_V032.csv')

# Get cooordinates of stations
stlat_list = station_df['Latitude'].values
stlon_list = station_df['Longitude'].values

# Find the indicies of the events within Bay Area geographic range
bay_stn_ind = np.where((stlat_list > 36) & (stlat_list < 39) &
                   (stlon_list > -123.5 ) & (stlon_list < -121))

# Get the names of the stations in the Bay Area
stn_list = station_df['Station Name'].values[bay_stn_ind]



#%%

# Histogram of magnitudes
plt.figure()
plt.hist(mag_list,edgecolor='k')
plt.xlabel('Magnitude')
plt.ylabel('Counts')
plt.savefig('/Users/tnye/bayarea_path/plots/NGA-West2_bayevents_maghist.png',dpi=300)
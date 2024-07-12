#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:27:24 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import os
import subprocess

# Read in gmprocess records file
records_df = pd.read_csv('/Users/tnye/bayarea_path/data/gmprocess/waveforms/data/bayarea_default_metrics_rotd(percentile=50.0).csv')

# Get list of unique stations
stns = np.unique(records_df['StationCode'])[1:] #for some reason the header was showing up in this array

# Depth array 
z_list = np.arange(0, -5000, -10)

# Needed to run geomodelgrids
os.chdir('/Users/tnye/code/geomodelgrids/')
result = subprocess.run("source ./setup.sh && env", shell=True, capture_output=True, text=True)

# Loop over stations
for stn in stns:
    
    # Initialize input file
    f = open(f'/Users/tnye/bayarea_path/files/velmod/station_Vs/{stn}_Vs.in','w')
    
    # Get coordinates of the station
    stn_idx = np.where(records_df['StationCode']==stn)[0][0]
    stlon = records_df['StationLongitude'].values[stn_idx]
    stlat = records_df['StationLatitude'].values[stn_idx]
    
    # Loop over depths
    for z in z_list:
        f.write(f'{stlat}\t{stlon}\t{z}\n')
    
    # Close input file
    f.close()
    
    # Query velocity model
    command = [
    "geomodelgrids_query",
    "--models=/Users/tnye/bayarea_path/files/velmod/USGS_SFCVM_v21-1_detailed.h5,/Users/tnye/bayarea_path/files/velmod/USGS_SFCVM_v21-0_regional.h5",
    f"--points=/Users/tnye/bayarea_path/files/velmod/station_Vs/{stn}_Vs.in",
    f"--output=/Users/tnye/bayarea_path/files/velmod/station_Vs/{stn}_Vs.out",
    "--values=Vs"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    
    


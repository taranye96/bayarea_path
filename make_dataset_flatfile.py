#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 13:35:00 2024

@author: tnye
"""

# Imports
from glob import glob
import pandas as pd
from gmprocess.io.asdf.stream_workspace import StreamWorkspace
from gmprocess.utils.constants import DATA_DIR

# Gather list of .h5 files
h5_files = sorted(glob('/Users/tnye/bayarea_path/data/gmprocess/waveforms/data/*/workspace.h5'))

# Initialize lists
evid_list = []
mag_list = []
hyplon_list = []
hyplat_list = []
hypdepth_list = []
stlon_list = []
stlat_list = []
stelev_list = []

# Loop over .h5 files
for i, file in enumerate(h5_files[:300]):
    
    # if i % 100 == 0:
    #     print(f'{300/i}% done')
    
    # Read in workspace
    workspace = StreamWorkspace.open(file)
    
    # Get eventID
    evid = workspace.get_event_ids()[0]

    # Loop over streams and read in the 3 components
    for st in workspace.get_streams(evid,labels=['default']):
    
        # Did the traces pass the processing tests? If so, add to flatfile
        if st[0].passed == True and st[1].passed == True:
    
            # Get station name and coordinates
            stn = st[0].stats.station
            stlon = st[0].stats.coordinates.longitude
            stlat = st[0].stats.coordinates.latitude
            stelev = st[0].stats.coordinates.elevation
            
            # Get event metadata
            event = workspace.get_event(evid)
            hyplon = event.origins[0].longitude
            hyplat = event.origins[0].latitude
            hypdepth = event.origins[0].depth
            mag = event.magnitudes[0].mag
            
            # Append data to lists
            evid_list.append(evid)
            mag_list.append(mag)
            hyplon_list.append(hyplon)
            hyplat_list.append(hyplat)
            hypdepth_list.append(hypdepth)
            stlon_list.append(stlon)
            stlat_list.append(stlat)
            stelev_list.append(stelev)
            
# Save to dataframe
data = {'Evid':evid_list,'Magnitude':mag_list,'Hyplon':hyplon_list,'Hyplat':hyplat_list,'Hypdepth':hypdepth_list,
        'Stlon':stlon_list,'Stlat':stlat_list,'Stelev':stelev_list}

df = pd.DataFrame.from_dict(data)

df.to_csv('/Users/tnye/bayarea_path/files/metadata/bayarea_downloaded_2000-2024.csv')
            
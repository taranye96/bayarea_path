#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:44:28 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import os
import subprocess

index = 2

directory = '/Users/tnye/bayarea_path/data/gmprocess/waveforms'
os.chdir(directory)    

# Read in event catalog (downlaoded from USGS latest earthquakes page)
event_catalog = pd.read_csv(f'/Users/tnye/bayarea_path/files/metadata/bayarea_catalog_2000-2024_libcomcat_{index}.csv')

#%%

evids = event_catalog['id'].values
hotstart_file = f'/Users/tnye/bayarea_path/data/gmprocess/event_files/bayevents_2000-2024_{index}_hotstart.txt'

if not os.path.isfile(hotstart_file):
    
    # Loop over event IDs
    for i, evid in enumerate(evids):
        
        f = open(f'/Users/tnye/bayarea_path/data/gmprocess/event_files/bayevents_2000-2024_{index}_hotstart.txt','w')
        f.write(f'{i}')
        f.close()
        
        # Download data
        subprocess.run(["gmrecords", "--eventid", evid, "download"])

else:
    
    # What event did we leave off on?
    hotstart = int(np.genfromtxt(hotstart_file))
    
    # Get event IDs
    evids = event_catalog['id'].values[hotstart:]
    
    # Loop over event IDs
    for evid in evids:
        
        f = open(f'/Users/tnye/bayarea_path/data/gmprocess/event_files/bayevents_2000-2024_{index}_hotstart.txt','w')
        f.write(f'{hotstart}')
        f.close()
        
        # Download data
        subprocess.run(["gmrecords", "--eventid", evid, "download"])
        
        hotstart +=1


#%%

# # File for saving evid index to come back to if download is interrupted
# hotstart_filea = f'/Users/tnye/bayarea_path/data/gmprocess/event_files/bayevents_2000-2024_{index}_hotstart.txt'
# hotstart_fileb = f'/Users/tnye/bayarea_path/data/gmprocess/event_files/bayevents_2000-2024_{index}b_hotstart.txt'

# # What event did we leave off on?
# hotstart_a = int(np.genfromtxt(hotstart_filea))
# hotstart_b = int(np.genfromtxt(hotstart_fileb))

# # Get event IDs
# evids = event_catalog['id'].values[hotstart_a+1:hotstart_b+1]
# evids = evids[::-1]

# # Loop over event IDs
# for evid in evids:
    
#     # Download data
#     subprocess.run(["gmrecords", "--eventid", evid, "download"])
    
#     hotstart_b -= 1
    
#     f = open(f'/Users/tnye/bayarea_path/data/gmprocess/event_files/bayevents_2000-2024_{index}b_hotstart.txt','w')
#     f.write(f'{hotstart_b}')
#     f.close()
    
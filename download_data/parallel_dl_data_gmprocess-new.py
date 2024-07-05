#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:59:57 2024

@author: tnye
"""


# Imports
import numpy as np
import pandas as pd
from glob import glob
from os import path, chdir
import subprocess
import shutil
from mpi4py import MPI

###################### Set up parallelization parameters ######################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

ncpus = size

# gather catalogs
catalogs = sorted(glob('/Users/tnye/bayarea_path/data/gmprocess/event_files/bayevents_2000-2024_*.csv'))

local_catalogs = len(catalogs) // ncpus
remainder = len(catalogs) % ncpus

if rank < remainder:
    local_catalogs +=1

start_index = rank * local_catalogs + min(rank, remainder)
end_index = start_index + local_catalogs

local_catalogs = catalogs[start_index:end_index]

local_catalog_sizes = np.array([len(local_catalogs)],dtype=int)
catalog_sizes = np.empty(size, dtype=int)
comm.Allgather([local_catalog_sizes, MPI.INT], [catalog_sizes, MPI.INT])

recvbuf = np.empty(local_catalog_sizes, dtype=object)
comm.Scatterv([local_catalogs, catalog_sizes, None, MPI.CHAR], recvbuf, root=0)

print(f'(Rank: {rank} received data={recvbuf}')


############################# Download data ###################################
    
directory = '/Users/tnye/bayarea_path/data/gmprocess/waveforms'
chdir(directory)     

for index in recvbuf:
    
    index = int(index)
    
    # Read in event catalog (downlaoded from USGS latest earthquakes page)
    event_catalog = pd.read_csv(f'/Users/tnye/bayarea_path/data/gmprocess/event_files/bayevents_2000-2024_{index}.csv')
    
    # Get event IDs
    evids = event_catalog['id'].values
    
    # File for saving evid index to come back to if download is interrupted
    hotstart_file = f'/Users/tnye/bayarea_path/data/gmprocess/event_files/bayevents_2000-2024_{index}_hotstart.txt'
    
    if not path.isfile(hotstart_file):
        
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
    


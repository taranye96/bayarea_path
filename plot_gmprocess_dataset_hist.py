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

# Read in NGA-West2 finite fault file
df = pd.read_csv('/Users/tnye/bayarea_path/data/gmprocess/bayarea_dataset_2000-2024.csv')

# Get the magnitudes of the events in the Bay Area
mag_list = df['Magnitude'].values

# Histogram of magnitudes
plt.figure()
plt.hist(mag_list,edgecolor='k')
plt.xlabel('Magnitude')
plt.ylabel('Counts')
plt.savefig('/Users/tnye/bayarea_path/plots/gmprocess_records.png',dpi=300)
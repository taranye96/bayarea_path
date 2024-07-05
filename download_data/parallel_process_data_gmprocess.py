#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:38:59 2024

@author: tnye
"""

###############################################################################
# This script performs a series of tasks on the data downloaded from gmprocess,
# which include processing the waveforms and computing metrics.
###############################################################################


# Imports
from os import chdir
import subprocess

# Chagne path to gmprocess directory
directory = '/Users/tnye/bayarea_path/data/gmprocess/waveforms'
chdir(directory)     

# Run gmprocess process commands
# subprocess.run(['gmrecords','assemble'])
subprocess.run(['gmrecords','process_waveforms'])
subprocess.run(['gmrecords','compute_station_metrics'])
subprocess.run(['gmrecords','compute_waveform_metrics'])

# there should be another command to export to excel file



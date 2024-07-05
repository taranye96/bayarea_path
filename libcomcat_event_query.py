#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:44:10 2024

@author: tnye
"""

###############################################################################
# Script used to find earthquake source info using libcomcat. 
###############################################################################

# Imports 
import numpy as np
import pandas as pd
from datetime import datetime
from libcomcat.search import search
from libcomcat.dataframes import get_detail_data_frame

events = search(starttime=datetime(2000, 1, 1),endtime=datetime(2024, 6, 2),minlatitude=36,maxlatitude=39,minlongitude=-123.5,
       maxlongitude=-121,minmagnitude=3)

df = get_detail_data_frame(events, get_all_magnitudes=True)

df.to_csv('/Users/tnye/bayarea_path/files/source_info/libcomcat_focals.csv')

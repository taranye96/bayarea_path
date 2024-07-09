#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:34:27 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt

# Read in kappa df from Nye et al. (2022)
Nye_kappa_df = pd.read_csv('/Users/tnye/bayarea_path/files/site_info/kappa/Nye22_BayArea_kappa.txt')

# Read in kappa df from Cabas et al. (2022)
Cabas_kappa_df = pd.read_csv('/Users/tnye/bayarea_path/files/site_info/kappa/DesignSafe_publish_k0_free.csv')

nye_stns = Nye_kappa_df['#Station']
cabas_stns = Cabas_kappa_df['StationCode']

joint_stns = np.intersect1d(nye_stns,cabas_stns)

nye_kappa = []
for i in range(len(Nye_kappa_df)):
    if Nye_kappa_df['#Station'].iloc[i] in joint_stns:
        nye_kappa.append(Nye_kappa_df['Kappa(s)'].iloc[i])
        
cabas_kappa = []
for stn in joint_stns:
    tmp_kappa = []
    idx = np.where(cabas_stns == stn)[0]
    for i in idx:
        tmp_kappa.append(Cabas_kappa_df['k0_sw'].iloc[i])
    cabas_kappa.append(gmean(tmp_kappa))

# Plot comparison
x = [0,0.1]
y = x

fig, ax = plt.subplots(1,1)
ax.plot(x,y,c='k')
ax.scatter(nye_kappa, cabas_kappa)
ax.set_xlabel(r'Nye et al. (2022) $\kappa_0$')
ax.set_ylabel(r'Cabas et al. (2022) $\kappa_0$')
plt.savefig('/Users/tnye/bayarea_path/plots/Nye_vs_Cabas_k0.png',dpi=300)



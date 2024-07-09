#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:30:48 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Which records were reiduals computed on?
res_dataset = 'all_records'
# res_dataset = '8evs_5stns'

# Read in path effects file
dS_df = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{res_dataset}/reformatted/SFBA_dS_meml.txt',delimiter='\t')

# Read in kappa values from Nye et al. (2022)
kappa_df = pd.read_csv('/Users/tnye/bayarea_path/files/site_info/kappa/DesignSafe_publish_k0_free.csv')

# Get list of stations from this dataset
all_stns = dS_df['Station'].values
stns = set(all_stns)
stns = sorted([str(value) for value in stns])

# Get list of kappa stations
kappa_stns = kappa_df['StationCode'].values

# Get list of kappa values
k0 = kappa_df['k0_sw'].values

# # Check that all kappa stations are in my dataset
# for kappa_stn in kappa_stns:
#     if kappa_stn not in stns:
#         print(kappa_stn)

# Get list of kappas and path effects for the overlapping stations in the datasets
k0_list = []
BSSA14_rock_dP_list = []
BSSA14_dP_list = []
ASK14_rock_dP_list = []
ASK14_dP_list = []
for i in range(len(dS_df)):
    stn = all_stns[i]
    if stn in kappa_stns:
        idx = np.where(kappa_stns == stn)[0][0]
        k0_list.append(k0[idx])
        BSSA14_rock_dP_list.append(dS_df['BSSA14rock_PGA_res dS'].iloc[i])
        BSSA14_dP_list.append(dS_df['BSSA14_PGA_res dS'].iloc[i])
        ASK14_rock_dP_list.append(dS_df['ASK14rock_PGA_res dS'].iloc[i])
        ASK14_dP_list.append(dS_df['ASK14_PGA_res dS'].iloc[i])


#%% Plot results

fig, axs = plt.subplots(2,2, figsize=(6,4.5))
axs[0,0].scatter(k0_list, BSSA14_rock_dP_list, c='palevioletred', s=5, alpha=0.5)
axs[0,0].set_title('BSSA14 Rock Conditions')
axs[0,0].grid(ls='--',alpha=0.5)

axs[0,1].scatter(k0_list, BSSA14_dP_list, c='palevioletred', s=5, alpha=0.5)
axs[0,1].set_title('BSSA14 Varying Site Conditions')
axs[0,1].grid(ls='--',alpha=0.5)

axs[1,0].scatter(k0_list, ASK14_rock_dP_list, c='palevioletred', s=5, alpha=0.5)
axs[1,0].set_title('ASK14 Rock Conditions')
axs[1,0].grid(ls='--',alpha=0.5)

axs[1,1].scatter(k0_list, ASK14_dP_list, c='palevioletred', s=5, alpha=0.5)
axs[1,1].set_title('ASK14 Varying Site Conditions')
axs[1,1].grid(ls='--',alpha=0.5)

plt.subplots_adjust(hspace=0.55, wspace=0.3)
fig.supylabel(r'$\delta$S')
fig.supxlabel(r'$\kappa_0$')

plt.savefig(f'/Users/tnye/bayarea_path/plots/k0_dS-{res_dataset}_DesignSafe.png',dpi=300)


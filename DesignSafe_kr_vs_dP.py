#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:07:27 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Which records were residuals computed on?
res_dataset = 'all_records'
# res_dataset = '8evs_5stns'

# Read in path effects file
dP_df = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{res_dataset}/reformatted/SFBA_dP_meml.txt',delimiter='\t')

# Read in kappa values from Nye et al. (2022)
kappa_df = pd.read_csv('/Users/tnye/bayarea_path/files/site_info/kappa/DesignSafe_publish_kr_free_band.csv')

# Get list of stations from this dataset
all_stns = dP_df['Station'].values
stns = set(all_stns)
stns = sorted([str(value) for value in stns])
events = np.unique(dP_df['Event'])

# Get list of kappa stations
kappa_stns = kappa_df['sta'].values
kappa_events = kappa_df['EarthquakeId'].values

# Get list of kappa values
kr_h1 = kappa_df['k_sw_h1'].values
kr_h2 = kappa_df['k_sw_h2'].values
kr = np.sqrt(kr_h1 * kr_h2)

missing_stn = []
for stn in stns:
    if stn not in kappa_stns:
        missing_stn.append(stn)

missing_event = []
for event in events:
    if event not in kappa_events:
        missing_event.append(event)



# Get list of kappas and path effects for the overlapping stations in the datasets
kr_list = []
BSSA14_rock_dP_list = []
BSSA14_dP_list = []
ASK14_rock_dP_list = []
ASK14_dP_list = []
for i in range(len(dP_df)):
    stn = all_stns[i]
    event = dP_df['Event'].iloc[i]
    if stn in kappa_stns:
        if event in kappa_events:
            idx = np.where(kappa_stns == stn)[0][0]
            kr_list.append(kr[idx])
            BSSA14_rock_dP_list.append(dP_df['BSSA14rock_PGA_res dP'].iloc[i])
            BSSA14_dP_list.append(dP_df['BSSA14_PGA_res dP'].iloc[i])
            ASK14_rock_dP_list.append(dP_df['ASK14rock_PGA_res dP'].iloc[i])
            ASK14_dP_list.append(dP_df['ASK14_PGA_res dP'].iloc[i])


#%% Plot results

fig, axs = plt.subplots(2,2, figsize=(6,4.5))
axs[0,0].scatter(kr_list, BSSA14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,0].set_title('BSSA14 Rock Conditions')
axs[0,0].grid(ls='--',alpha=0.5)

axs[0,1].scatter(kr_list, BSSA14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,1].set_title('BSSA14 Varying Site Conditions')
axs[0,1].grid(ls='--',alpha=0.5)

axs[1,0].scatter(kr_list, ASK14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,0].set_title('ASK14 Rock Conditions')
axs[1,0].grid(ls='--',alpha=0.5)

axs[1,1].scatter(kr_list, ASK14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,1].set_title('ASK14 Varying Site Conditions')
axs[1,1].grid(ls='--',alpha=0.5)

plt.subplots_adjust(hspace=0.55, wspace=0.3)
fig.supylabel(r'$\delta$P')
fig.supxlabel(r'$\kappa_R$')

plt.savefig(f'/Users/tnye/bayarea_path/plots/kr_dP-{res_dataset}-DesignSafe.png',dpi=300)


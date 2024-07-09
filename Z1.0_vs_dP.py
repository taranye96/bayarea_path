#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:51:45 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

min_evs = 10
min_stns = 10

res_dataset = f'{min_evs}evs_{min_stns}stns'

# Read in path effects file
records_df = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/GMM_residuals_culled-{min_evs}evs-{min_stns}stns.csv')
dP_df = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{res_dataset}/reformatted/SFBA_dP_meml.txt',delimiter='\t')

# Read in kappa values from Nye et al. (2022)
z1_df = pd.read_csv('/Users/tnye/bayarea_path/files/site_info/station_z1.0.csv')

# Get list of stations from this dataset
all_stns = dP_df['Station'].values
stns = set(all_stns)
stns = sorted([str(value) for value in stns])

# Get list of kappa stations
z1_stns = z1_df['Station'].values

# Get list of kappa values
z1 = z1_df['Z1.0(m)'].values

rrup = records_df['Rrup(km)']

# Get list of kappas and path effects for the overlapping stations in the datasets
z1_list = []
BSSA14_rock_dP_list = []
BSSA14_dP_list = []
ASK14_rock_dP_list = []
ASK14_dP_list = []
rrup_list = []
lon = []
lat = []
z1_perc = []
for i in range(len(dP_df)):
    stn = all_stns[i]
    if stn in z1_stns:
        idx = np.where(z1_stns == stn)[0][0]
        z1_list.append(z1[idx])
        z1_perc.append(z1[idx]/rrup[i])
        BSSA14_rock_dP_list.append(dP_df['BSSA14rock_PGA_res dP'].iloc[i])
        BSSA14_dP_list.append(dP_df['BSSA14_PGA_res dP'].iloc[i])
        ASK14_rock_dP_list.append(dP_df['ASK14rock_PGA_res dP'].iloc[i])
        ASK14_dP_list.append(dP_df['ASK14_PGA_res dP'].iloc[i])
        rrup_list.append(rrup[i])

#%% Plot results

fig, axs = plt.subplots(2,2, figsize=(6,4.5))
axs[0,0].scatter(z1_list, BSSA14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,0].set_title('BSSA14 Rock Conditions')
axs[0,0].grid(ls='--',alpha=0.5)

axs[0,1].scatter(z1_list, BSSA14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,1].set_title('BSSA14 Varying Site Conditions')
axs[0,1].grid(ls='--',alpha=0.5)

axs[1,0].scatter(z1_list, ASK14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,0].set_title('ASK14 Rock Conditions')
axs[1,0].grid(ls='--',alpha=0.5)

axs[1,1].scatter(z1_list, ASK14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,1].set_title('ASK14 Varying Site Conditions')
axs[1,1].grid(ls='--',alpha=0.5)

plt.subplots_adjust(hspace=0.55, wspace=0.3)
fig.supylabel(r'$\delta$P')
fig.supxlabel('Z1.0 (m)')

# plt.savefig('/Users/tnye/bayarea_path/plots/Z1.0_dP.png',dpi=300)


#%% Plot dP as a function of Z1 percentage

fig, axs = plt.subplots(2,2, figsize=(6,4.5))
axs[0,0].scatter(np.array(z1_perc)/1000, BSSA14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,0].set_title('BSSA14 Rock Conditions')
axs[0,0].grid(ls='--',alpha=0.5)
# axs[0,0].set_xscale('log')

axs[0,1].scatter(np.array(z1_perc)/1000, BSSA14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,1].set_title('BSSA14 Varying Site Conditions')
axs[0,1].grid(ls='--',alpha=0.5)

axs[1,0].scatter(np.array(z1_perc)/1000, ASK14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,0].set_title('ASK14 Rock Conditions')
axs[1,0].grid(ls='--',alpha=0.5)

axs[1,1].scatter(np.array(z1_perc)/1000, ASK14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,1].set_title('ASK14 Varying Site Conditions')
axs[1,1].grid(ls='--',alpha=0.5)
# axs[1,1].set_xscale('log')

plt.subplots_adjust(hspace=0.55, wspace=0.3)
fig.supylabel(r'$\delta$P')
fig.supxlabel('Z1.0/rrup')

plt.savefig(f'/Users/tnye/bayarea_path/plots/Z1.0perc_vs_dP-{res_dataset}.png',dpi=300)



#%%

vmin = -6.5
vmax = 6.5

fig, axs = plt.subplots(2,2, figsize=(6,5.75))
sc = axs[0,0].scatter(rrup_list, z1_list, c=BSSA14_rock_dP_list, cmap='seismic', vmin=vmin, vmax=vmax, s=7, alpha=0.7)
axs[0,0].set_title('BSSA14 Rock Conditions')
axs[0,0].grid(ls='--',alpha=0.5)

axs[0,1].scatter(rrup_list, z1_list, c=BSSA14_dP_list, cmap='seismic', vmin=vmin, vmax=vmax, s=7, alpha=0.7)
axs[0,1].set_title('BSSA14 Varying Site Conditions')
axs[0,1].grid(ls='--',alpha=0.5)

axs[1,0].scatter(rrup_list, z1_list, c=ASK14_rock_dP_list, cmap='seismic', vmin=vmin, vmax=vmax, s=7, alpha=0.7)
axs[1,0].set_title('ASK14 Rock Conditions')
axs[1,0].grid(ls='--',alpha=0.5)

axs[1,1].scatter(rrup_list, z1_list, c=ASK14_dP_list, cmap='seismic', vmin=vmin, vmax=vmax, s=7, alpha=0.7)
axs[1,1].set_title('ASK14 Varying Site Conditions')
axs[1,1].grid(ls='--',alpha=0.5)

cbar = fig.colorbar(sc, ax=[axs[1,0], axs[1,1]], orientation='horizontal', pad=1)
cbar.set_label(r'$\delta$P')

plt.subplots_adjust(hspace=0.55, wspace=0.4, bottom =0.25)
fig.supylabel('Z1.0 (m)')
fig.supxlabel('Rrup (km)')


#%%

import pygmt

# Specify the map region and projection
region = [-123.5, -121, 36, 39]
projection = 'M6i'

# Begin figure
fig = pygmt.Figure()

# Plot basemap with coastlines and borders
fig.coast(region=region, projection=projection, shorelines=True, borders=1, frame=True)

# Plot stations colored by value
fig.plot(x=z1_df['Longitude'], y=z1_df['Latitude'], style='c0.15c', color=z1, cmap='viridis', scale=True)

# Add colorbar
fig.colorbar(position='JMR', frame=['a1000', 'x+l"Value"'])

# Save or show the plot
fig.show()
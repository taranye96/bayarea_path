#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:07:27 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Which records were residuals computed on?
# res_dataset = 'all_records'
res_dataset = '5evs_5stns'

# Read in path effects file
records_df = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/GMM_residuals_culled.csv')
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
        

rrup = records_df['Rrup(km)']

# Get list of kappas and path effects for the overlapping stations in the datasets
kr_list = np.array([])
BSSA14_rock_dP_list = np.array([])
BSSA14_dP_list = np.array([])
ASK14_rock_dP_list = np.array([])
ASK14_dP_list = np.array([])
rrup_list = np.array([])
for i in range(len(dP_df)):
    stn = all_stns[i]
    event = dP_df['Event'].iloc[i]
    if stn in kappa_stns:
        if event in kappa_events:
            idx = np.where(kappa_stns == stn)[0][0]
            kr_list = np.append(kr_list, kr[idx])
            BSSA14_rock_dP_list = np.append(BSSA14_rock_dP_list, dP_df['BSSA14rock_PGA_res dP'].iloc[i])
            BSSA14_dP_list = np.append(BSSA14_dP_list, dP_df['BSSA14_PGA_res dP'].iloc[i])
            ASK14_rock_dP_list = np.append(ASK14_rock_dP_list,dP_df['ASK14rock_PGA_res dP'].iloc[i])
            ASK14_dP_list = np.append(ASK14_dP_list, dP_df['ASK14_PGA_res dP'].iloc[i])
            rrup_list = np.append(rrup_list, rrup[i])

pga_list = np.array([])
pgv_list = np.array([])
sa01_list = np.array([])
sa1_list = np.array([])
sa10_list = np.array([])
for i in range(len(dP_df)):
    stn = all_stns[i]
    event = dP_df['Event'].iloc[i]
    if stn in kappa_stns:
        if event in kappa_events:
            pga_list = np.append(pga_list, dP_df['ASK14_PGA_res dP'].iloc[i])
            pgv_list = np.append(pgv_list, dP_df['ASK14_PGV_res dP'].iloc[i])
            sa01_list = np.append(sa01_list, dP_df['ASK14_SA_res_T0.01 dP'].iloc[i])
            sa1_list = np.append(sa1_list, dP_df['ASK14_SA_res_T1.0 dP'].iloc[i])
            sa10_list = np.append(sa10_list, dP_df['ASK14_SA_res_T10.0 dP'].iloc[i])



#%% Plot results

rrup_lims = [0, 50, 100, 150, 200]
nbins = [10, 10,10,8,8]

for i, rrup_lim in enumerate(rrup_lims):
    idx = np.where(rrup_list >= rrup_lim)[0]
    
    x = kr_list
    x_reg = x.reshape(-1, 1) 
    
    bin_edges = np.linspace(0,0.12,nbins[i])
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    pga_bin_means, _, _ = stats.binned_statistic(x[idx],pga_list[idx],statistic='mean', bins=bin_edges)
    pga_bin_std, _, _ = stats.binned_statistic(x[idx],pga_list[idx],statistic='std', bins=bin_edges)
    
    pgv_bin_means, _, _ = stats.binned_statistic(x[idx],pgv_list[idx],statistic='mean', bins=bin_edges)
    pgv_bin_std, _, _ = stats.binned_statistic(x[idx],pgv_list[idx],statistic='std', bins=bin_edges)
    
    sa01_bin_means, _, _ = stats.binned_statistic(x[idx],sa01_list[idx],statistic='mean', bins=bin_edges)
    sa01_bin_std, _, _ = stats.binned_statistic(x[idx],sa01_list[idx],statistic='std', bins=bin_edges)
    
    sa1_bin_means, _, _ = stats.binned_statistic(x[idx],sa1_list[idx],statistic='mean', bins=bin_edges)
    sa1_bin_std, _, _ = stats.binned_statistic(x[idx],sa1_list[idx],statistic='std', bins=bin_edges)
    
    sa10_bin_means, _, _ = stats.binned_statistic(x[idx],sa10_list[idx],statistic='mean', bins=bin_edges)
    sa10_bin_std, _, _ = stats.binned_statistic(x[idx],sa10_list[idx],statistic='std', bins=bin_edges)
    
    fig, axs = plt.subplots(2, 3, figsize=(7,4.5))
    axs[0,2].remove()
    
    axs[0,0].scatter(kr_list[idx], pga_list[idx], facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
    axs[0,0].errorbar(bin_centers, pga_bin_means, yerr=pga_bin_std, markersize=1.5, elinewidth=0.75, color='k', fmt="o", label=r'Bin mean and $1\sigma$')
    axs[0,0].set_title('PGA')
    axs[0,0].grid(ls='--',alpha=0.5)
    
    axs[0,1].scatter(kr_list[idx], pgv_list[idx], facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
    axs[0,1].errorbar(bin_centers, pgv_bin_means, yerr=pga_bin_std, markersize=1.5, elinewidth=0.75, color='k', fmt="o", label=r'Bin mean and $1\sigma$')
    axs[0,1].set_title('PGV')
    axs[0,1].grid(ls='--',alpha=0.5)
    
    axs[1,0].scatter(kr_list[idx], sa01_list[idx], facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
    axs[1,0].errorbar(bin_centers, sa01_bin_means, yerr=pga_bin_std, markersize=1.5, elinewidth=0.75, color='k', fmt="o", label=r'Bin mean and $1\sigma$')
    axs[1,0].set_title('SA T=0.1 s')
    axs[1,0].grid(ls='--',alpha=0.5)
    
    axs[1,1].scatter(kr_list[idx], sa1_list[idx], facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
    axs[1,1].errorbar(bin_centers, sa1_bin_means, yerr=pga_bin_std, markersize=1.5, elinewidth=0.75, color='k', fmt="o", label=r'Bin mean and $1\sigma$')
    axs[1,1].set_title('SA T=1 s')
    axs[1,1].grid(ls='--',alpha=0.5)
    
    axs[1,2].scatter(kr_list[idx], sa10_list[idx], facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
    axs[1,2].errorbar(bin_centers, sa10_bin_means, yerr=pga_bin_std, markersize=1.5, elinewidth=0.75, color='k', fmt="o", label=r'Bin mean and $1\sigma$')
    axs[1,2].set_title('SA T=10 s')
    axs[1,2].grid(ls='--',alpha=0.5)
    
    plt.subplots_adjust(hspace=0.55, wspace=0.3)
    fig.supylabel(r'$\delta$P')
    fig.supxlabel(r'$\kappa_R$')
    plt.show()
    
    if rrup_lim == 0:
        plt.savefig('/Users/tnye/bayarea_path/plots/kr_plots/kr-DesignSafe_dP.png',dpi=300)
    else:
        plt.savefig(f'/Users/tnye/bayarea_path/plots/kr_plots/kr-DesignSafe_dP_{rrup_lim}km.png',dpi=300)
    

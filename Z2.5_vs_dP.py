#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:51:45 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

min_evs = 5
min_stns = 5

res_dataset = f'{min_evs}evs_{min_stns}stns'

# Read in path effects file
records_df = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/GMM_residuals_culled.csv')
dP_df = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{res_dataset}/reformatted/SFBA_dP_meml.txt',delimiter='\t')

# Read in kappa values from Nye et al. (2022)
z25_df = pd.read_csv('/Users/tnye/bayarea_path/files/site_info/station_z2.5.csv')

# Get list of stations from this dataset
all_stns = dP_df['Station'].values
stns = set(all_stns)
stns = sorted([str(value) for value in stns])

# Get list of kappa stations
z25_stns = z25_df['Station'].values

# Get list of kappa values
z25 = z25_df['Z1.0(m)'].values

rrup = records_df['Rrup(km)']

# Get list of kappas and path effects for the overlapping stations in the datasets
z25_list = np.array([])
BSSA14_rock_dP_list = np.array([])
BSSA14_dP_list = np.array([])
ASK14_rock_dP_list = np.array([])
ASK14_dP_list = np.array([])
rrup_list = np.array([])
lon = np.array([])
lat = np.array([])
z25_perc = np.array([])
for i in range(len(dP_df)):
    stn = all_stns[i]
    if stn in z25_stns:
        idx = np.where(z25_stns == stn)[0][0]
        if z25[idx] == 0:
            z25[idx] = 0.001
        z25_list = np.append(z25_list, z25[idx])
        z25_perc = np.append(z25_perc, z25[idx]/rrup[i])
        BSSA14_rock_dP_list = np.append(BSSA14_rock_dP_list, dP_df['BSSA14rock_PGV_res dP'].iloc[i])
        BSSA14_dP_list = np.append(BSSA14_dP_list, dP_df['BSSA14_PGV_res dP'].iloc[i])
        ASK14_rock_dP_list = np.append(ASK14_rock_dP_list, dP_df['ASK14rock_PGV_res dP'].iloc[i])
        ASK14_dP_list = np.append(ASK14_dP_list, dP_df['ASK14_PGV_res dP'].iloc[i])
        rrup_list = np.append(rrup_list, rrup[i])

pga_list = np.array([])
pgv_list = np.array([])
sa01_list = np.array([])
sa1_list = np.array([])
sa10_list = np.array([])
for i in range(len(dP_df)):
    stn = all_stns[i]
    if stn in z25_stns:
        pga_list = np.append(pga_list, dP_df['ASK14_PGA_res dP'].iloc[i])
        pgv_list = np.append(pgv_list, dP_df['ASK14_PGV_res dP'].iloc[i])
        sa01_list = np.append(sa01_list, dP_df['ASK14_SA_res_T0.01 dP'].iloc[i])
        sa1_list = np.append(sa1_list, dP_df['ASK14_SA_res_T1.0 dP'].iloc[i])
        sa10_list = np.append(sa10_list, dP_df['ASK14_SA_res_T10.0 dP'].iloc[i])



#%% Plot dP as a function of Z1 percentage

fig, axs = plt.subplots(2,2, figsize=(6,4.5))
axs[0,0].scatter(np.array(z25_perc)/1000, BSSA14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,0].set_title('BSSA14 Rock Conditions')
axs[0,0].grid(ls='--',alpha=0.5)
# axs[0,0].set_xscale('log')

axs[0,1].scatter(np.array(z25_perc)/1000, BSSA14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,1].set_title('BSSA14 Varying Site Conditions')
axs[0,1].grid(ls='--',alpha=0.5)

axs[1,0].scatter(np.array(z25_perc)/1000, ASK14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,0].set_title('ASK14 Rock Conditions')
axs[1,0].grid(ls='--',alpha=0.5)

axs[1,1].scatter(np.array(z25_perc)/1000, ASK14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,1].set_title('ASK14 Varying Site Conditions')
axs[1,1].grid(ls='--',alpha=0.5)
# axs[1,1].set_xscale('log')

plt.subplots_adjust(hspace=0.55, wspace=0.3)
fig.supylabel(r'$\delta$P')
fig.supxlabel('Z2.5/rrup')

plt.savefig(f'/Users/tnye/bayarea_path/plots/Z_plots/Z2.5perc_vs_dP.png',dpi=300)


#%% Plot dP as a function of Z1 percentage

idx = np.where(np.array(z25_perc)/1000 >= 0)[0]
x = (np.array(z25_perc)[idx]/1000).reshape(-1, 1) 

bssa14rock_regr = LinearRegression() 
bssa14rock_regr.fit(x, np.array(BSSA14_rock_dP_list)[idx].reshape(-1, 1)) 
y_bssa14rock = bssa14rock_regr.predict(x) 

bssa14_regr = LinearRegression() 
bssa14_regr.fit(x, np.array(BSSA14_dP_list)[idx].reshape(-1, 1)) 
bssa14_regr.predict(x) 
y_bssa14 = bssa14_regr.predict(x) 


fig, axs = plt.subplots(2,2, figsize=(6,4.5))
axs[0,0].scatter(np.array(z25_perc)/1000, BSSA14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,0].plot(x, y_bssa14rock, c='k', lw=1)
axs[0,0].set_title('BSSA14 Rock Conditions')
axs[0,0].grid(ls='--',alpha=0.5)
# axs[0,0].set_xscale('log')

axs[0,1].scatter(np.array(z25_perc)/1000, BSSA14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,1].plot(x, y_bssa14, c='k', lw=1)
axs[0,1].set_title('BSSA14 Varying Site Conditions')
axs[0,1].grid(ls='--',alpha=0.5)

axs[1,0].scatter(np.array(z25_perc)/1000, ASK14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,0].set_title('ASK14 Rock Conditions')
axs[1,0].grid(ls='--',alpha=0.5)

axs[1,1].scatter(np.array(z25_perc)/1000, ASK14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,1].set_title('ASK14 Varying Site Conditions')
axs[1,1].grid(ls='--',alpha=0.5)
# axs[1,1].set_xscale('log')

plt.subplots_adjust(hspace=0.55, wspace=0.3)
fig.supylabel(r'$\delta$P')
fig.supxlabel('Z2.5/rrup')

plt.savefig(f'/Users/tnye/bayarea_path/plots/Z_plots/Z2.5perc_vs_dP_regr.png',dpi=300)


#%% Plot results

fig, axs = plt.subplots(2,2, figsize=(6,4.5))
axs[0,0].scatter(z25_list, BSSA14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,0].set_title('BSSA14 Rock Conditions')
axs[0,0].grid(ls='--',alpha=0.5)

axs[0,1].scatter(z25_list, BSSA14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[0,1].set_title('BSSA14 Varying Site Conditions')
axs[0,1].grid(ls='--',alpha=0.5)

axs[1,0].scatter(z25_list, ASK14_rock_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,0].set_title('ASK14 Rock Conditions')
axs[1,0].grid(ls='--',alpha=0.5)

axs[1,1].scatter(z25_list, ASK14_dP_list, facecolors='none', edgecolors='palevioletred', s=5, alpha=0.1)
axs[1,1].set_title('ASK14 Varying Site Conditions')
axs[1,1].grid(ls='--',alpha=0.5)

plt.subplots_adjust(hspace=0.55, wspace=0.3)
fig.supylabel(r'$\delta$P')
fig.supxlabel('Z2.5 (m)')

plt.savefig('/Users/tnye/bayarea_path/plots/Z_plots/Z2.5_dP.png',dpi=300)


#%%

vmin = -6.5
vmax = 6.5

fig, axs = plt.subplots(2,2, figsize=(6,5.75))
sc = axs[0,0].scatter(rrup_list, z25_list, c=BSSA14_rock_dP_list, cmap='seismic', vmin=vmin, vmax=vmax, s=7, alpha=0.7)
axs[0,0].set_title('BSSA14 Rock Conditions')
axs[0,0].grid(ls='--',alpha=0.5)

axs[0,1].scatter(rrup_list, z25_list, c=BSSA14_dP_list, cmap='seismic', vmin=vmin, vmax=vmax, s=7, alpha=0.7)
axs[0,1].set_title('BSSA14 Varying Site Conditions')
axs[0,1].grid(ls='--',alpha=0.5)

axs[1,0].scatter(rrup_list, z25_list, c=ASK14_rock_dP_list, cmap='seismic', vmin=vmin, vmax=vmax, s=7, alpha=0.7)
axs[1,0].set_title('ASK14 Rock Conditions')
axs[1,0].grid(ls='--',alpha=0.5)

axs[1,1].scatter(rrup_list, z25_list, c=ASK14_dP_list, cmap='seismic', vmin=vmin, vmax=vmax, s=7, alpha=0.7)
axs[1,1].set_title('ASK14 Varying Site Conditions')
axs[1,1].grid(ls='--',alpha=0.5)

cbar = fig.colorbar(sc, ax=[axs[1,0], axs[1,1]], orientation='horizontal', pad=1)
cbar.set_label(r'$\delta$P')

plt.subplots_adjust(hspace=0.55, wspace=0.4, bottom =0.25)
fig.supylabel('Z2.5 (m)')
fig.supxlabel('Rrup (km)')

plt.savefig(f'/Users/tnye/bayarea_path/plots/Z_plots/Rrup_vs_Z2.5.png',dpi=300)



#%% Z2.5 % of Rrup for the diffrent IMs

x = z25_perc/1000
x_reg = x.reshape(-1, 1) 

z25perc_bin_edges = np.linspace(0,0.84,10)
bin_centers = (z25perc_bin_edges[1:] + z25perc_bin_edges[:-1]) / 2

pga_bin_means, _, _ = stats.binned_statistic(x,pga_list,statistic='mean', bins=z25perc_bin_edges)
pga_bin_std, _, _ = stats.binned_statistic(x,pga_list,statistic='std', bins=z25perc_bin_edges)

pgv_bin_means, _, _ = stats.binned_statistic(x,pgv_list,statistic='mean', bins=z25perc_bin_edges)
pgv_bin_std, _, _ = stats.binned_statistic(x,pgv_list,statistic='std', bins=z25perc_bin_edges)

sa01_bin_means, _, _ = stats.binned_statistic(x,sa01_list,statistic='mean', bins=z25perc_bin_edges)
sa01_bin_std, _, _ = stats.binned_statistic(x,sa01_list,statistic='std', bins=z25perc_bin_edges)

sa1_bin_means, _, _ = stats.binned_statistic(x,sa1_list,statistic='mean', bins=z25perc_bin_edges)
sa1_bin_std, _, _ = stats.binned_statistic(x,sa1_list,statistic='std', bins=z25perc_bin_edges)

sa10_bin_means, _, _ = stats.binned_statistic(x,sa10_list,statistic='mean', bins=z25perc_bin_edges)
sa10_bin_std, _, _ = stats.binned_statistic(x,sa10_list,statistic='std', bins=z25perc_bin_edges)

pga_regr = LinearRegression() 
pga_regr.fit(x_reg, pga_list.reshape(-1, 1)) 
y_pga = pga_regr.predict(x_reg) 

pgv_regr = LinearRegression() 
pgv_regr.fit(x_reg, pgv_list.reshape(-1, 1)) 
y_pgv = pgv_regr.predict(x_reg) 

sa01_regr = LinearRegression() 
sa01_regr.fit(x_reg, sa01_list.reshape(-1, 1)) 
y_sa01 = sa01_regr.predict(x_reg) 

sa1_regr = LinearRegression() 
sa1_regr.fit(x_reg, sa1_list.reshape(-1, 1)) 
y_sa1 = sa1_regr.predict(x_reg) 

sa10_regr = LinearRegression() 
sa10_regr.fit(x_reg, sa10_list.reshape(-1, 1)) 
y_sa10 = sa10_regr.predict(x_reg) 

fig, axs = plt.subplots(2, 3, figsize=(7,4.5))
axs[0,2].remove()

axs[0,0].scatter(x, pga_list, c=rrup_list, cmap='viridis', vmin=0, vmax=320, s=5, alpha=0.1)
axs[0,0].errorbar(bin_centers, pga_bin_means, yerr=pga_bin_std, markersize=1.5, elinewidth=0.75, color='goldenrod', fmt="o", label=r'Bin mean and $1\sigma$')
# axs[0,0].plot(x, y_pga, c='k', lw=1)
axs[0,0].set_title('PGA')
axs[0,0].grid(ls='--',alpha=0.5)
# axs[0,0].set_xscale('log')

sc = axs[0,1].scatter(x, pgv_list, c=rrup_list, cmap='viridis', vmin=0, vmax=320,  s=5, alpha=0.1)
axs[0,1].errorbar(bin_centers, pgv_bin_means, yerr=pgv_bin_std, markersize=1.5, elinewidth=0.75, color='goldenrod', fmt="o", label=r'Bin mean and $1\sigma$')
# axs[0,1].plot(x, y_pgv, c='k', lw=1)
axs[0,1].set_title('PGV')
axs[0,1].grid(ls='--',alpha=0.5)

axs[1,0].scatter(x, sa01_list, c=rrup_list, cmap='viridis', vmin=0, vmax=320,  s=5, alpha=0.1)
axs[1,0].errorbar(bin_centers, sa01_bin_means, yerr=sa01_bin_std, markersize=1.5, elinewidth=0.75, color='goldenrod', fmt="o", label=r'Bin mean and $1\sigma$')
# axs[1,0].plot(x, y_sa01, c='k')
axs[1,0].set_title('SA T=0.01 s')
axs[1,0].grid(ls='--',alpha=0.5)

axs[1,1].scatter(x, sa1_list, c=rrup_list, cmap='viridis', vmin=0, vmax=320,  s=5, alpha=0.1)
axs[1,1].errorbar(bin_centers, sa1_bin_means, yerr=sa1_bin_std, markersize=1.5, elinewidth=0.75, color='goldenrod', fmt="o", label=r'Bin mean and $1\sigma$')
# axs[1,0].plot(x, y_sa01, c='k')
axs[1,1].set_title('SA T=1 s')
axs[1,1].grid(ls='--',alpha=0.5)

axs[1,2].scatter(x, sa10_list, c=rrup_list, cmap='viridis', vmin=0, vmax=320,  s=5, alpha=0.1)
axs[1,2].errorbar(bin_centers, sa10_bin_means, yerr=sa10_bin_std, markersize=1.5, elinewidth=0.75, color='goldenrod', fmt="o", label=r'Bin mean and $1\sigma$')
# axs[1,0].plot(x, y_sa01, c='k')
axs[1,2].set_title('SA T=10 s')
axs[1,2].grid(ls='--',alpha=0.5)

# Create a ScalarMappable for the colorbar
norm = mcolors.Normalize(vmin=0, vmax=320)
cmap = cm.get_cmap('viridis')
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Create a colorbar axis
cbar_ax = fig.add_axes([0.17, 0.1, 0.675, 0.03])

# Add the colorbar with the specified orientation
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label(r'$R_{rup}$')

plt.subplots_adjust(hspace=0.5, wspace=0.4, bottom=0.25, top=0.925)
fig.supylabel(r'$\delta$P-MEML')
axs[1,1].set_xlabel(r'$Z_{2.5}$/$R_{rup}$ (km)',fontsize='large')

# fig.supxlabel('Z2.5/rrup')

plt.savefig(f'/Users/tnye/bayarea_path/plots/Z_plots/Z2.5perc_vs_dP_IM_regr.png',dpi=300)



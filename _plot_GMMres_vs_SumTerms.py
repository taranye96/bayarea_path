#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:57:22 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Which records were reiduals computed on?
min_evs = 1
min_stns = 1

records_df_1_1 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/GMM_residuals_culled-{min_evs}evs-{min_stns}stns.csv')
dS_df_1_1 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{min_evs}evs_{min_stns}stns/reformatted/SFBA_dS_meml.txt',delimiter='\t')
dP_df_1_1 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{min_evs}evs_{min_stns}stns/reformatted/SFBA_dP_meml.txt',delimiter='\t')
dE_df_1_1 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{min_evs}evs_{min_stns}stns/reformatted/SFBA_dE_meml.txt',delimiter='\t')

dS_1_1 = dS_df_1_1['ASK14_PGA_res dS'].values
dP_1_1 = dP_df_1_1['ASK14_PGA_res dP'].values
dE_1_1 = dE_df_1_1['ASK14_PGA_res dE'].values
PGAres_1_1 = records_df_1_1['ASK14_PGA_res']
sum_1_1 = dS_1_1 + dP_1_1 + dE_1_1

# Which records were reiduals computed on?
min_evs = 10
min_stns = 10

records_df_10_10 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/GMM_residuals_culled-{min_evs}evs-{min_stns}stns.csv')
dS_df_10_10 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{min_evs}evs_{min_stns}stns/reformatted/SFBA_dS_meml.txt',delimiter='\t')
dP_df_10_10 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{min_evs}evs_{min_stns}stns/reformatted/SFBA_dP_meml.txt',delimiter='\t')
dE_df_10_10 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{min_evs}evs_{min_stns}stns/reformatted/SFBA_dE_meml.txt',delimiter='\t')

dS_10_10 = dS_df_10_10['ASK14_PGA_res dS'].values
dP_10_10 = dP_df_10_10['ASK14_PGA_res dP'].values
dE_10_10 = dE_df_10_10['ASK14_PGA_res dE'].values
PGAres_10_10 = records_df_10_10['ASK14_PGA_res']
sum_10_10 = dS_10_10 + dP_10_10 + dE_10_10

# Which records were reiduals computed on?
min_evs = 100
min_stns = 100

records_df_100_100 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/GMM_residuals_culled-{min_evs}evs-{min_stns}stns.csv')
dS_df_100_100 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{min_evs}evs_{min_stns}stns/reformatted/SFBA_dS_meml.txt',delimiter='\t')
dP_df_100_100 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{min_evs}evs_{min_stns}stns/reformatted/SFBA_dP_meml.txt',delimiter='\t')
dE_df_100_100 = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{min_evs}evs_{min_stns}stns/reformatted/SFBA_dE_meml.txt',delimiter='\t')

dS_100_100 = dS_df_100_100['ASK14_PGA_res dS'].values
dP_100_100 = dP_df_100_100['ASK14_PGA_res dP'].values
dE_100_100 = dE_df_100_100['ASK14_PGA_res dE'].values
PGAres_100_100 = records_df_100_100['ASK14_PGA_res']
sum_100_100 = dS_100_100 + dP_100_100 + dE_100_100

#%% 

x = [-7, 10]
y = x

fig, axs = plt.subplots(2,2, figsize=(5,4))
axs[1,1].remove()

axs[0,0].plot(x, y, c='k')
axs[0,0].scatter(PGAres_1_1, sum_1_1,  c='cornflowerblue', linewidths=0.5, alpha=0.3)
axs[0,0].grid(alpha=0.5)
axs[0,0].set_xlim(-7,10)
axs[0,0].set_ylim(-7,10)
# axs[0,0].set_aspect('equal', adjustable='box')
axs[0,0].set_title('Min 1 event/station')

axs[0,1].plot(x, y, c='k')
axs[0,1].scatter(PGAres_10_10, sum_10_10, c='cornflowerblue', linewidths=0.5, alpha=0.3)
axs[0,1].grid(alpha=0.5)
axs[0,1].set_xlim(-7,10)
axs[0,1].set_xlim(-7,10)
# axs[0,1].set_aspect('equal', adjustable='box')
axs[0,1].set_title('Min 10 events/stations')

axs[1,0].plot(x, y, c='k')
axs[1,0].scatter(PGAres_100_100, sum_100_100, c='cornflowerblue', linewidths=0.5, alpha=0.3)
axs[1,0].grid(alpha=0.5)
axs[1,0].set_xlim(-7,10)
axs[1,0].set_xlim(-7,10)
# axs[1,0].set_aspect('equal', adjustable='box')
axs[1,0].set_title('Min 100 events/stations')

fig.supxlabel('GMM PGA residual')
fig.supylabel(r'$\delta$E + $\delta$P + $\delta$S')
# axs[0,0].set_ylabel(r'$\delta$E')
# axs[0,1].set_ylabel(r'$\delta$P')
# axs[1,0].set_ylabel(r'$\delta$S')

plt.subplots_adjust(wspace=0.5,hspace=0.5)

plt.savefig('/Users/tnye/bayarea_path/plots/GMM-res_decomp-res.png',dpi=300)

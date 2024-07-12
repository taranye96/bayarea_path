#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:33:00 2024

@author: tnye
"""

###############################################################################
# This script computes the residuals between observed and estimated ground 
# motion for the Bay Area path effects study. 
###############################################################################

# Imports
import numpy as np
import pandas as pd

min_stns = 5
min_evs = 5

outfile = '/Users/tnye/bayarea_path/files/residual_analysis/GMM_residuals_culled.csv'

# Read in IM flatfiles
bssa14_rock_df = pd.read_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_BSSA14_rock.csv')
bssa14_df = pd.read_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_BSSA14.csv')
ask14_rock_df = pd.read_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_ASK14_rock.csv')
ask14_df = pd.read_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_ASK14.csv')


#%% Cull records to sations that have recorded at least 8 events and events
  # That have been recorded on at least 5 stations

stn_idx = np.array([])
ev_idx = np.array([])

events = np.unique(bssa14_df['Event'])
stns = np.unique(bssa14_df['Station'])

for stn in stns:
    idx = np.where(ask14_df['Station'] == stn)[0]
    if len(idx) >= min_evs:
        stn_idx = np.append(stn_idx,idx)
    
for ev in events:
    idx = np.where(ask14_df['Event'] == ev)[0]
    if len(idx) >= min_stns:
        ev_idx = np.append(ev_idx,idx)

useable_idx = np.intersect1d(stn_idx,ev_idx)

bssa14_rock_df = bssa14_rock_df.loc[useable_idx].reset_index()
bssa14_df = bssa14_df.loc[useable_idx].reset_index()
ask14_rock_df = ask14_rock_df.loc[useable_idx].reset_index()
ask14_df = ask14_df.loc[useable_idx].reset_index()


#%%

# Get obseved IMs
obs_pga = bssa14_df['Obs_PGA(g)'].values
obs_pgv = bssa14_df['Obs_PGV(cm/s)'].values
obs_SA = bssa14_df.loc[:, 'Obs_SA(T=0.01)(g)':'Obs_SA(T=10.0)(g)'].values

# Get estimated IMs from GMMs
bssa14_pga_rock = bssa14_rock_df['BSSA14_PGA(g)'].values
bssa14_pgv_rock = bssa14_rock_df['BSSA14_PGV(cm/s)'].values
bssa14_SA_rock = bssa14_rock_df.loc[:, 'BSSA14_SA(T=0.01)(g)':'BSSA14_SA(T=10.0)(g)'].values
bssa14_pga = bssa14_df['BSSA14_PGA(g)'].values
bssa14_pgv = bssa14_df['BSSA14_PGV(cm/s)'].values
bssa14_SA = bssa14_df.loc[:, 'BSSA14_SA(T=0.01)(g)':'BSSA14_SA(T=10.0)(g)'].values
ask14_pga_rock = ask14_rock_df['ASK14_PGA(g)'].values
ask14_pgv_rock = ask14_rock_df['ASK14_PGV(cm/s)'].values
ask14_SA_rock = ask14_rock_df.loc[:, 'ASK14_SA(T=0.01)(g)':'ASK14_SA(T=10.0)(g)'].values
ask14_pga = ask14_df['ASK14_PGA(g)'].values
ask14_pgv = ask14_df['ASK14_PGV(cm/s)'].values
ask14_SA = ask14_df.loc[:, 'ASK14_SA(T=0.01)(g)':'ASK14_SA(T=10.0)(g)'].values

# Calc GMM residuals
bssa14_pga_rock_res = np.log(np.array(obs_pga)/np.array(bssa14_pga_rock))
bssa14_pga_res = np.log(np.array(obs_pga)/np.array(bssa14_pga))
bssa14_pgv_rock_res = np.log(np.array(obs_pgv)/np.array(bssa14_pgv_rock))
bssa14_pgv_res = np.log(np.array(obs_pgv)/np.array(bssa14_pgv))
bssa14_SA_rock_res = np.log(np.array(obs_SA)/np.array(bssa14_SA_rock))
bssa14_SA_res = np.log(np.array(obs_SA)/np.array(bssa14_SA))
ask14_pga_rock_res = np.log(np.array(obs_pga)/np.array(ask14_pga_rock))
ask14_pga_res = np.log(np.array(obs_pga)/np.array(ask14_pga))
ask14_pgv_rock_res = np.log(np.array(obs_pgv)/np.array(ask14_pgv_rock))
ask14_pgv_res = np.log(np.array(obs_pgv)/np.array(ask14_pgv))
ask14_SA_rock_res = np.log(np.array(obs_SA)/np.array(ask14_SA_rock))
ask14_SA_res = np.log(np.array(obs_SA)/np.array(ask14_SA))


#%% Set up residual dataframe

# Parameter and observed IM columns fro the IM dataframes
main_df = bssa14_df.loc[:,'Event':'Obs_SA(T=10.0)(g)']

# Peak ground motion residual columns
pgm_res_dict = {'BSSA14rock_PGA_res':bssa14_pga_rock_res,'BSSA14_PGA_res':bssa14_pga_res,
        'ASK14rock_PGA_res':ask14_pga_rock_res,'ASK14_PGA_res':ask14_pga_res,
        'BSSA14rock_PGV_res':bssa14_pgv_rock_res,'BSSA14_PGV_res':bssa14_pgv_res,
        'ASK14rock_PGV_res':ask14_pgv_rock_res,'ASK14_PGV_res':ask14_pgv_res}
pgm_res_df = pd.DataFrame(pgm_res_dict)

# BSSA14 SA rock residual columns
bssa14rock_SA_col_headers = ['BSSA14rock_SA_res_T0.01','BSSA14rock_SA_res_T0.02','BSSA14rock_SA_res_T0.03',
                  'BSSA14rock_SA_res_T0.05','BSSA14rock_SA_res_T0.075','BSSA14rock_SA_res_T0.1',
                  'BSSA14rock_SA_res_T0.15','BSSA14rock_SA_res_T0.2','BSSA14rock_SA_res_T0.25',
                  'BSSA14rock_SA_res_T0.3','BSSA14rock_SA_res_T0.4','BSSA14rock_SA_res_T0.5',
                  'BSSA14rock_SA_res_T0.75','BSSA14rock_SA_res_T1.0','BSSA14rock_SA_res_T1.5',
                  'BSSA14rock_SA_res_T2.0','BSSA14rock_SA_res_T3.0','BSSA14rock_SA_res_T4.0',
                  'BSSA14rock_SA_res_T5.0','BSSA14rock_SA_res_T7.5','BSSA14rock_SA_res_T10.0']
bssa14rock_SA_df = pd.DataFrame(bssa14_SA_rock_res, columns=bssa14rock_SA_col_headers)
     
# BSSA14 SA residual columns           
bssa14_SA_col_headers = ['BSSA14_SA_res_T0.01','BSSA14_SA_res_T0.02','BSSA14_SA_res_T0.03',
                  'BSSA14_SA_res_T0.05','BSSA14_SA_res_T0.075','BSSA14_SA_res_T0.1',
                  'BSSA14_SA_res_T0.15','BSSA14_SA_res_T0.2','BSSA14_SA_res_T0.25',
                  'BSSA14_SA_res_T0.3','BSSA14_SA_res_T0.4','BSSA14_SA_res_T0.5',
                  'BSSA14_SA_res_T0.75','BSSA14_SA_res_T1.0','BSSA14_SA_res_T1.5',
                  'BSSA14_SA_res_T2.0','BSSA14_SA_res_T3.0','BSSA14_SA_res_T4.0',
                  'BSSA14_SA_res_T5.0','BSSA14_SA_res_T7.5','BSSA14_SA_res_T10.0']
bssa14_SA_df = pd.DataFrame(bssa14_SA_res, columns=bssa14_SA_col_headers)

# ASK14 SA rock residual columns
ask14rock_SA_col_headers = ['ASK14rock_SA_res_T0.01','ASK14rock_SA_res_T0.02','ASK14rock_SA_res_T0.03',
                  'ASK14rock_SA_res_T0.05','ASK14rock_SA_res_T0.075','ASK14rock_SA_res_T0.1',
                  'ASK14rock_SA_res_T0.15','ASK14rock_SA_res_T0.2','ASK14rock_SA_res_T0.25',
                  'ASK14rock_SA_res_T0.3','ASK14rock_SA_res_T0.4','ASK14rock_SA_res_T0.5',
                  'ASK14rock_SA_res_T0.75','ASK14rock_SA_res_T1.0','ASK14rock_SA_res_T1.5',
                  'ASK14rock_SA_res_T2.0','ASK14rock_SA_res_T3.0','ASK14rock_SA_res_T4.0',
                  'ASK14rock_SA_res_T5.0','ASK14rock_SA_res_T7.5','ASK14rock_SA_res_T10.0']
ask14rock_SA_df = pd.DataFrame(ask14_SA_rock_res, columns=ask14rock_SA_col_headers)

# ASK14 SA residual columns
ask14_SA_col_headers = ['ASK14_SA_res_T0.01','ASK14_SA_res_T0.02','ASK14_SA_res_T0.03',
                  'ASK14_SA_res_T0.05','ASK14_SA_res_T0.075','ASK14_SA_res_T0.1',
                  'ASK14_SA_res_T0.15','ASK14_SA_res_T0.2','ASK14_SA_res_T0.25',
                  'ASK14_SA_res_T0.3','ASK14_SA_res_T0.4','ASK14_SA_res_T0.5',
                  'ASK14_SA_res_T0.75','ASK14_SA_res_T1.0','ASK14_SA_res_T1.5',
                  'ASK14_SA_res_T2.0','ASK14_SA_res_T3.0','ASK14_SA_res_T4.0',
                  'ASK14_SA_res_T5.0','ASK14_SA_res_T7.5','ASK14_SA_res_T10.0']
ask14_SA_df = pd.DataFrame(ask14_SA_res, columns=ask14_SA_col_headers)

# Final dataframe
res_df = main_df.join(pgm_res_df.join(bssa14rock_SA_df.join(bssa14_SA_df.join(ask14rock_SA_df.join(ask14_SA_df)))))
res_df.to_csv(outfile)


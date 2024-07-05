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

# Read in IM flatfiles
bssa14_rock_df = pd.read_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_BSSA14_rock.csv')
bssa14_df = pd.read_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_BSSA14.csv')
ask14_rock_df = pd.read_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_ASK14_rock.csv')
ask14_df = pd.read_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_ASK14.csv')

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
bssa14rock_SA_col_headers = ['BSSA14rock_SA_res(T=0.01)(g)','BSSA14rock_SA_res(T=0.02)(g)','BSSA14rock_SA_res(T=0.03)(g)',
                  'BSSA14rock_SA_res(T=0.05)(g)','BSSA14rock_SA_res(T=0.075)(g)','BSSA14rock_SA_res(T=0.1)(g)',
                  'BSSA14rock_SA_res(T=0.15)(g)','BSSA14rock_SA_res(T=0.2)(g)','BSSA14rock_SA_res(T=0.25)(g)',
                  'BSSA14rock_SA_res(T=0.3)(g)','BSSA14rock_SA_res(T=0.4)(g)','BSSA14rock_SA_res(T=0.5)(g)',
                  'BSSA14rock_SA_res(T=0.75)(g)','BSSA14rock_SA_res(T=1.0)(g)','BSSA14rock_SA_res(T=1.5)(g)',
                  'BSSA14rock_SA_res(T=2.0)(g)','BSSA14rock_SA_res(T=3.0)(g)','BSSA14rock_SA_res(T=4.0)(g)',
                  'BSSA14rock_SA_res(T=5.0)(g)','BSSA14rock_SA_res(T=7.5)(g)','BSSA14rock_SA_res(T=10.0)(g)']
bssa14rock_SA_df = pd.DataFrame(bssa14_SA_rock_res, columns=bssa14rock_SA_col_headers)
     
# BSSA14 SA residual columns           
bssa14_SA_col_headers = ['BSSA14_SA_res(T=0.01)(g)','BSSA14_SA_res(T=0.02)(g)','BSSA14_SA_res(T=0.03)(g)',
                  'BSSA14_SA_res(T=0.05)(g)','BSSA14_SA_res(T=0.075)(g)','BSSA14_SA_res(T=0.1)(g)',
                  'BSSA14_SA_res(T=0.15)(g)','BSSA14_SA_res(T=0.2)(g)','BSSA14_SA_res(T=0.25)(g)',
                  'BSSA14_SA_res(T=0.3)(g)','BSSA14_SA_res(T=0.4)(g)','BSSA14_SA_res(T=0.5)(g)',
                  'BSSA14_SA_res(T=0.75)(g)','BSSA14_SA_res(T=1.0)(g)','BSSA14_SA_res(T=1.5)(g)',
                  'BSSA14_SA_res(T=2.0)(g)','BSSA14_SA_res(T=3.0)(g)','BSSA14_SA_res(T=4.0)(g)',
                  'BSSA14_SA_res(T=5.0)(g)','BSSA14_SA_res(T=7.5)(g)','BSSA14_SA_res(T=10.0)(g)']
bssa14_SA_df = pd.DataFrame(bssa14_SA_res, columns=bssa14_SA_col_headers)

# ASK14 SA rock residual columns
ask14rock_SA_col_headers = ['ASK14rock_SA_res(T=0.01)(g)','ASK14rock_SA_res(T=0.02)(g)','ASK14rock_SA_res(T=0.03)(g)',
                  'ASK14rock_SA_res(T=0.05)(g)','ASK14rock_SA_res(T=0.075)(g)','ASK14rock_SA_res(T=0.1)(g)',
                  'ASK14rock_SA_res(T=0.15)(g)','ASK14rock_SA_res(T=0.2)(g)','ASK14rock_SA_res(T=0.25)(g)',
                  'ASK14rock_SA_res(T=0.3)(g)','ASK14rock_SA_res(T=0.4)(g)','ASK14rock_SA_res(T=0.5)(g)',
                  'ASK14rock_SA_res(T=0.75)(g)','ASK14rock_SA_res(T=1.0)(g)','ASK14rock_SA_res(T=1.5)(g)',
                  'ASK14rock_SA_res(T=2.0)(g)','ASK14rock_SA_res(T=3.0)(g)','ASK14rock_SA_res(T=4.0)(g)',
                  'ASK14rock_SA_res(T=5.0)(g)','ASK14rock_SA_res(T=7.5)(g)','ASK14rock_SA_res(T=10.0)(g)']
ask14rock_SA_df = pd.DataFrame(ask14_SA_rock_res, columns=ask14rock_SA_col_headers)

# ASK14 SA residual columns
ask14_SA_col_headers = ['ASK14_SA_res(T=0.01)(g)','ASK14_SA_res(T=0.02)(g)','ASK14_SA_res(T=0.03)(g)',
                  'ASK14_SA_res(T=0.05)(g)','ASK14_SA_res(T=0.075)(g)','ASK14_SA_res(T=0.1)(g)',
                  'ASK14_SA_res(T=0.15)(g)','ASK14_SA_res(T=0.2)(g)','ASK14_SA_res(T=0.25)(g)',
                  'ASK14_SA_res(T=0.3)(g)','ASK14_SA_res(T=0.4)(g)','ASK14_SA_res(T=0.5)(g)',
                  'ASK14_SA_res(T=0.75)(g)','ASK14_SA_res(T=1.0)(g)','ASK14_SA_res(T=1.5)(g)',
                  'ASK14_SA_res(T=2.0)(g)','ASK14_SA_res(T=3.0)(g)','ASK14_SA_res(T=4.0)(g)',
                  'ASK14_SA_res(T=5.0)(g)','ASK14_SA_res(T=7.5)(g)','ASK14_SA_res(T=10.0)(g)']
ask14_SA_df = pd.DataFrame(ask14_SA_res, columns=ask14_SA_col_headers)

# Final dataframe
res_df = main_df.join(pgm_res_df.join(bssa14rock_SA_df.join(bssa14_SA_df.join(ask14rock_SA_df.join(ask14_SA_df)))))
res_df.to_csv('/Users/tnye/bayarea_path/files/residual_analysis/GMM_residuals.csv')

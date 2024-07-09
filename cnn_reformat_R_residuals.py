#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:14:54 2020

@author: vjs
"""

###############################################################################
# Script that reformats the output of the residual decomposition
# (cnn_compute_meml_residuals.R) into text documents for each residual type. 
###############################################################################

import numpy as np
import pandas as pd

min_evs = 100
min_stns = 100

project_directory = f'/Users/tnye/bayarea_path/files/residual_analysis/R_MEML/{min_evs}evs_{min_stns}stns'

## location:
loc_name = 'SFBA'

## which model:
model_name_list = ['BSSA14rock_PGA_res','BSSA14_PGA_res','ASK14rock_PGA_res','ASK14_PGA_res']

## Read in the main file with pandas:
res_df = pd.read_csv(f'/Users/tnye/bayarea_path/files/residual_analysis/GMM_residuals_culled-{min_evs}evs-{min_stns}stns.csv')

## Make empty dataframes for the event and site, with just event and site info:
event_df = res_df.copy()[['Event',  'Qlon', 'Qlat', 'Qdepth', 'Magnitude']]
site_df = res_df.copy()[['Station', 'Slon', 'Slat', 'Selev']]
path_df = res_df.copy()[['Event',  'Qlon', 'Qlat', 'Qdepth', 'Magnitude',
                         'Station', 'Slon', 'Slat', 'Selev']]

for i_model in range(len(model_name_list)):
    i_model_name = model_name_list[i_model]
    i_model_filepath_list = i_model_name.split(" ")
    i_model_filepath = ".".join(i_model_filepath_list)
    print(i_model_filepath)
    
    ## Initiate empty event and site arrays for this model to fill
    i_event_residual = np.zeros(len(res_df))
    i_event_sd = np.zeros(len(res_df))
    
    i_site_residual = np.zeros(len(res_df))
    i_site_sd = np.zeros(len(res_df))
    
    i_path_residual = np.zeros(len(res_df))
    i_path_sd = np.zeros(len(res_df))
        
    ## Read in the event, and site files that correspond to this:
    event_file = f'{project_directory}/R_output/{loc_name}_event_{i_model_filepath}.txt'
    site_file = f'{project_directory}/R_output/{loc_name}_site_{i_model_filepath}.txt'
    fixed_file = f'{project_directory}/R_output/{loc_name}_fixed_{i_model_filepath}.txt'
    
    ## read in
    event_data = pd.read_csv(event_file)
    site_data = pd.read_csv(site_file)
    fixed_data = pd.read_csv(fixed_file)
    
    ## For each event in the event df, find the event indices in the main df
    for i_event in range(len(event_data)):
        i_event_id = event_data['ID'][i_event]
        i_event_indices = np.where(res_df['Event'] == i_event_id)[0]
        
        i_event_residual[i_event_indices] = event_data.Bias[i_event]
        i_event_sd[i_event_indices] = event_data['Std.error'][i_event]
        
    ## For each site in the site df, find the site indices in teh main df
    for i_site in range(len(site_data)):
        i_site_id = site_data['ID'][i_site]
        i_site_indices = np.where(res_df['Station'] == i_site_id)[0]
        
        i_site_residual[i_site_indices] = site_data.Bias[i_site]
        i_site_sd[i_site_indices] = site_data['Std.error'][i_site]
        
    ## Then, obtain the path residual - total minus bias minus event, minus site:
    i_path_residual = res_df[i_model_name] - fixed_data['Estimate'][0] - i_event_residual - i_site_residual 
    
    ## Add to the df:
    ## get column names for dataframe - event:
    event_col_name = i_model_name + ' dE'
    eventsd_col_name = i_model_name + ' dE std'
    
    ## site:
    site_col_name = i_model_name + ' dS'
    sitesd_col_name = i_model_name + ' dS std'
    
    ## path:
    path_col_name = i_model_name + ' dP'
    
    ## add to total df:
    res_df[event_col_name] = i_event_residual
    res_df[eventsd_col_name] = i_event_sd
    
    res_df[site_col_name] = i_site_residual
    res_df[sitesd_col_name] = i_site_sd
    
    res_df[path_col_name] = i_path_residual
    
    ## add event info to event df:
    event_df[event_col_name] = i_event_residual
    event_df[eventsd_col_name] = i_event_sd
    
    ## add site info to site df:
    site_df[site_col_name] = i_site_residual
    site_df[sitesd_col_name] = i_site_sd
    
    ## add path info to path df:
    path_df[path_col_name] = i_path_residual
    # path_df[pathsd_col_name] = i_path_sd
        

## Write out the dataframe:
output_path = f'{project_directory}/reformatted/{loc_name}_residuals_meml.txt'
res_df.to_csv(output_path,index=False,sep='\t')

# make and write out event, path, and site dataframes:
event_filepath =  f'{project_directory}/reformatted/{loc_name}_dE_meml.txt'
event_df.to_csv(event_filepath,index=False,sep='\t')

site_filepath =  f'{project_directory}/reformatted/{loc_name}_dS_meml.txt'
site_df.to_csv(site_filepath,index=False,sep='\t') 

path_filepath =  f'{project_directory}/reformatted/{loc_name}_dP_meml.txt'
path_df.to_csv(path_filepath,index=False,sep='\t') 

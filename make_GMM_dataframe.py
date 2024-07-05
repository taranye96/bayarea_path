#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:33:29 2024

@author: tnye
"""

###############################################################################
# This script creates flatfiles with the observed and estimated ground motions
# for the Bay Area path effects study. 
###############################################################################

# Imports
import numpy as np
import pandas as pd
import baypath_main_fns as main_fns

# Metrics flatfile from gmprocess for the processed records
records_df = pd.read_csv('/Users/tnye/bayarea_path/data/gmprocess/waveforms/data/bayarea_default_metrics_rotd(percentile=50.0).csv')

step_size = len(records_df) // 10

# Waveforms directory
wf_dir = '/Users/tnye/bayarea_path/data/gmprocess/waveforms/data'


#%% Get parameters needed for GMM

# Z1.0 data frame
z1_df = pd.read_csv('/Users/tnye/bayarea_path/files/site_info/station_z1.0.csv')
z1_stns = z1_df['Station'].values

# Vs30 dataframe from Thompson 2022 model
vs30_df = pd.read_csv('/Users/tnye/bayarea_path/files/site_info/vs30/bayarea_station_vs30.csv')

# Measured Vs30 dataframe fro McPhillips 2020
meas_vs30_df = pd.read_csv('/Users/tnye/bayarea_path/files/site_info/vs30/VS30_mcphillips_2020.csv')
meas_vs30_list = np.array(meas_vs30_df['VS30__M_S_'])
meas_vs30_stns = np.array(meas_vs30_df['NETWORK_ST'])
for i in range(len(meas_vs30_stns)):
    try:
        meas_vs30_stns[i] = meas_vs30_stns[i].split('.')[1]
    except:
        continue
    
# Libcomcat query source info
focal_df = pd.read_csv('/Users/tnye/bayarea_path/files/source_info/libcomcat_focals.csv')

# NGA-West 2 source info
nga_events_df = pd.read_csv('/Users/tnye/bayarea_path/files/NGA_West2_supporting_data_for_flatfile/NGA_West2_Finite_Fault_Info_050142013.csv')

# NGA-West 2 site catalog
nga_stns_df = pd.read_csv('/Users/tnye/bayarea_path/files/NGA_West2_supporting_data_for_flatfile/NGA_West2_SiteDatabase_V032.csv')

names = []
events = []
origins = []
Qlon = []
Qlat = []
Qdep = []
mag_list = []
rhyp_list = []
rrup_list = []
rjb_list = []
Slon = []
Slat = []
elev = []
vs30_meas_list = []
vs30_list = []
ztor_list = []
rake_list = []
dip_list = []
width_list = []
vs30_list = []
vs30_meas_list = []
z1pt0_list = []
obs_pga = []
obs_pgv = []
obs_SA = []
bssa14_pga_rock = []
bssa14_pgv_rock = []
bssa14_SA_rock = []
bssa14_pga = []
bssa14_pgv = []
bssa14_SA = []
ask14_pga_rock = []
ask14_pgv_rock = []
ask14_SA_rock = []
ask14_pga = []
ask14_pgv = []
ask14_SA = []
bssa14_pga_std_rock = []
bssa14_pgv_std_rock = []
bssa14_SA_std_rock = []
bssa14_pga_std = []
bssa14_pgv_std = []
bssa14_SA_std = []
ask14_pga_std_rock = []
ask14_pgv_std_rock = []
ask14_SA_std_rock = []
ask14_pga_std = []
ask14_pgv_std = []
ask14_SA_std = []

# Loop over records in dataframe
for i in range(len(records_df)):
    
    if i % step_size == 0:
        percentage_complete = (i / len(records_df)) * 100
        print(f"{percentage_complete:.0f}% complete")
    
    # Get basic event and site info
    quake = records_df['EarthquakeId'].iloc[i]
    stn = records_df['StationCode'].iloc[i]
    
    # Only use stations in desired study area (a few were slightly out of bounds)
    if stn in z1_stns:
        
        
        ########################### Get observed IMs ##########################
        
        # Get obs IMs
        obs_pga.append(records_df['PGA'].values[i]/100)
        obs_pgv.append(records_df['PGV'].values[i])
        obs_SA.append([records_df['SA(T=0.0100, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.0200, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.0300, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.0500, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.0750, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.1000, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.1500, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.2000, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.2500, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.3000, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.4000, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.5000, D=0.050)'].values[i]/100,
                        records_df['SA(T=0.7500, D=0.050)'].values[i]/100,
                        records_df['SA(T=1.0000, D=0.050)'].values[i]/100,
                        records_df['SA(T=1.5000, D=0.050)'].values[i]/100,
                        records_df['SA(T=2.0000, D=0.050)'].values[i]/100,
                        records_df['SA(T=3.0000, D=0.050)'].values[i]/100,
                        records_df['SA(T=4.0000, D=0.050)'].values[i]/100,
                        records_df['SA(T=5.0000, D=0.050)'].values[i]/100,
                        records_df['SA(T=7.5000, D=0.050)'].values[i]/100,
                        records_df['SA(T=10.0000, D=0.050)'].values[i]/100])
        
        
        ######################## Set up GMM parameters ########################
    
        names.append(stn)
        events.append(quake)
        mag = records_df['EarthquakeMagnitude'].iloc[i]
        rhyp = records_df['HypocentralDistance'].iloc[i]
        rrup = records_df['RuptureDistance'].iloc[i]
        rjb = records_df['JoynerBooreDistance'].iloc[i]
        if rrup >= rjb:
            ztor = np.sqrt(rrup**2 - rjb**2)
        else:
            ztor = 0
        
        Qlon.append(records_df['EarthquakeLongitude'].iloc[i])
        Qlat.append(records_df['EarthquakeLatitude'].iloc[i])
        Qdep.append(records_df['EarthquakeDepth'].iloc[i])
        origins.append(records_df['EarthquakeTime'].iloc[i])
        Slon.append(records_df['StationLongitude'].iloc[i])
        Slat.append(records_df['StationLatitude'].iloc[i])
        elev.append(records_df['StationElevation'].iloc[i])
        mag_list.append(mag)
        rhyp_list.append(rhyp)
        rrup_list.append(rhyp)
        rjb_list.append(rjb)
        ztor_list.append(ztor)
        
        # Rake, dip, width
        if quake[2:] in np.array(nga_events_df['Earthquake Name']):
            ind = np.where(nga_events_df['Earthquake Name']==quake[2:])[0][0]
            rake = nga_events_df['Rake (deg)'].iloc[ind]
            dip = nga_events_df['Dip  (deg)'].iloc[ind]
            width = nga_events_df['Width (km)'].iloc[ind]
        elif quake in np.array(focal_df['id']):
            ind = np.where(focal_df['id']==quake)[0][0]
            rake = focal_df['nc_np1_rake'].iloc[ind]
            dip = focal_df['nc_np1_dip'].iloc[ind]
            w = 'calc'
        else:
            rake = 0
            dip = 90
            w = 'calc'
        # Get rid of faulty data (some had rake, dip = -999)
        if np.abs(rake)>180:
                rake = 0
                dip = 90
        
        if w == 'calc':
            mag = records_df['EarthquakeMagnitude'].iloc[i]
            if mag < 5:
                # Wells and Coppersmith 1984 scaling law
                stress = 5*10**6
                M0 = 10**((3/2)*mag + 9.1)
                width = np.cbrt((7/16)*M0/stress)/1000
            else:
                # Thingbaijam scaling law for reverse earthquakes
                b = 0.435
                a = -1.669
                width = 10**(a + (b*mag))/1000
    
        width_list.append(width)
        rake_list.append(rake)
        dip_list.append(dip)
        
        # Vs30
        # # # Check if measured values exist:
        # if stn in meas_vs30_stns:
        #     stn_idx = np.where(meas_vs30_stns==stn)[0][0]
        #     v = meas_vs30_list[stn_idx]
        #     vs30_meas.append(v)
        #     vs30.append(v)
        
        # # Check if in NGA flatfile
        # elif stn in np.array(nga_stns_df['Station ID']):
        #     ind = np.where(nga_stns_df['Station ID']==stn)[0][0]
        #     v = nga_stns_df['Vs30 for Analysis (m/s)'].iloc[ind]
        #     vs30.append(v)
        #     if np.isnan(float(nga_stns_df['Measured  Vs30 (m/s) when zp > 30 m; inferred from Vsz otherwise '].iloc[ind])):
        #         vs30_meas.append(0.0)
        #     else:
        #         vs30_meas.append(v)
            
        # # Else get from global grid file
        # else:
        vs30 = round(vs30_df['Vs30(m/s)'].iloc[np.where(vs30_df['Station']==stn)[0][0]],3)
        vs30_meas = False
        vs30_list.append(vs30)
        vs30_meas_list.append('False')

        # Get Z1.0
        z1_ind = np.where(z1_df['Station']==stn)[0][0]
        z1 = z1_df['Z1.0(m)'].values[z1_ind]
        z1pt0_list.append(z1)
        
        
        ######################### Calc GMM estimates  #########################
        
        # Periods from gmprocess metric file
        periods = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                    0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
       
        # For BSSA14
        vs30_ref = 760.0
        z1_ref = 1000 #I didn't see a reference value in their paper, so I've listed 1km
       
        # Get BSSA14 estimates no site
        lnpga, pga_std = main_fns.bssa14('PGA',mag,rrup,rjb,ztor,rake,vs30_ref,vs30_meas,z1_ref)
        pga = np.exp(lnpga)
        bssa14_pga_rock.append(np.abs(pga[0]))
        bssa14_pga_std_rock.append(pga_std[0][0])
        
        lnpgv, pgv_std = main_fns.bssa14('PGV',mag,rrup,rjb,ztor,rake,vs30_ref,vs30_meas,z1_ref)
        pgv = np.exp(lnpgv) 
        bssa14_pgv_rock.append(np.abs(pgv[0]))
        bssa14_pgv_std_rock.append(pgv_std[0][0])
        
        sa_list = []
        sa_std_list = []
        for period in periods:
            lnSA, SA_std = main_fns.bssa14('SA',mag,rrup,rjb,ztor,rake,vs30_ref,vs30_meas,z1_ref,period=period)
            SA = np.exp(lnSA)
            sa_list.append(np.abs(SA[0]))
            sa_std_list.append(SA_std[0][0])
        bssa14_SA_rock.append(sa_list)
        bssa14_SA_std_rock.append(sa_std_list)
        
        # Get BSSA14 estimates with site
        lnpga, pga_std = main_fns.bssa14('PGA',mag,rrup,rjb,ztor,rake,vs30,vs30_meas,z1)
        pga = np.exp(lnpga)
        bssa14_pga.append(np.abs(pga[0]))
        bssa14_pga_std.append(pga_std[0][0])
        
        lnpgv, pgv_std = main_fns.bssa14('PGV',mag,rrup,rjb,ztor,rake,vs30,vs30_meas,z1)
        pgv = np.exp(pgv)
        bssa14_pgv.append(np.abs(pgv[0]))
        bssa14_pgv_std.append(pgv_std[0][0])
        
        sa_list = []
        sa_std_list = []
        for period in periods:
            lnSA, SA_std = main_fns.bssa14('SA',mag,rrup,rjb,ztor,rake,vs30,vs30_meas,z1,period=period)
            SA = np.exp(lnSA)
            sa_list.append(np.abs(SA[0]))
            sa_std_list.append(SA_std[0][0])
        bssa14_SA.append(sa_list)
        bssa14_SA_std.append(sa_std_list)
        
        # For ASK14
        vs30_ref = 1180
        z1_ref = 0.003 #from z1.0_ref equation in paper
       
        # Get ASK14 estimates no site
        lnpga, pga_std = main_fns.ask14('PGA',mag,rrup,rjb,ztor,rake,dip,width,vs30_ref,vs30_meas,z1_ref)
        pga = np.exp(lnpga)
        ask14_pga_rock.append(np.abs(pga[0]))
        ask14_pga_std_rock.append(pga_std[0][0])
        
        lnpgv, pgv_std = main_fns.ask14('PGV',mag,rrup,rjb,ztor,rake,dip,width,vs30_ref,vs30_meas,z1_ref)
        pgv = np.exp(lnpgv)
        ask14_pgv_rock.append(np.abs(pgv[0]))
        ask14_pgv_std_rock.append(pgv_std[0][0])
        
        sa_list = []
        sa_std_list = []
        for period in periods:
            lnSA, SA_std = main_fns.ask14('SA',mag,rrup,rjb,ztor,rake,dip,width,vs30_ref,vs30_meas,z1_ref,period=period)
            SA = np.exp(lnSA)
            sa_list.append(np.abs(SA[0]))
            sa_std_list.append(SA_std[0][0])
        ask14_SA_rock.append(sa_list)
        ask14_SA_std_rock.append(sa_std_list)
        
        # Get ASK14 estimates with site
        lnpga, pga_std = main_fns.ask14('PGA',mag,rrup,rjb,ztor,rake,dip,width,vs30,vs30_meas,z1)
        pga = np.exp(lnpga)
        ask14_pga.append(np.abs(pga[0]))
        ask14_pga_std.append(pga_std[0][0])

        lnpgv, pgv_std = main_fns.ask14('PGV',mag,rrup,rjb,ztor,rake,dip,width,vs30,vs30_meas,z1)
        pgv = np.exp(pgv)
        ask14_pgv.append(np.abs(pgv[0]))
        ask14_pgv_std.append(pgv_std[0][0])
        
        sa_list = []
        sa_std_list = []
        for period in periods:
            lnSA, SA_std = main_fns.ask14('SA',mag,rrup,rjb,ztor,rake,dip,width,vs30,vs30_meas,z1,period=period)
            SA = np.exp(lnSA)
            sa_list.append(np.abs(SA[0]))
            sa_std_list.append(SA_std[0][0])
        ask14_SA.append(sa_list)
        ask14_SA_std.append(sa_std_list)
        



#%% Calc GMM residuals

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


#%% Create first part of dataframe

# Create dictionary 
param_dict = {'Event':events,'Magnitude':mag_list,'Qlon':Qlon,'Qlat':Qlat,'Qdepth':Qdep,
                'Station':names,'Slon':Slon,'Slat':Slat,'Selev':elev,'Rhyp(km)':rhyp_list,
                'Rrup(km)':rrup_list,'Rjb(km)':rjb_list,'Ztor(km)':ztor_list,'Rake':rake_list,
                'Dip':dip_list,'Width(km)':width_list,'Vs30(m/s)':vs30_list,'Vs30_meas':vs30_meas_list,
                'Z1.0(m)':z1pt0_list,'Obs_PGA(g)':obs_pga,'Obs_PGV(cm/s)':obs_pgv}
param_df = pd.DataFrame(param_dict)

SA_col_headers = ['Obs_SA(T=0.01)(g)','Obs_SA(T=0.02)(g)','Obs_SA(T=0.03)(g)',
                  'Obs_SA(T=0.05)(g)','Obs_SA(T=0.075)(g)','Obs_SA(T=0.1)(g)',
                  'Obs_SA(T=0.15)(g)','Obs_SA(T=0.2)(g)','Obs_SA(T=0.25)(g)',
                  'Obs_SA(T=0.3)(g)','Obs_SA(T=0.4)(g)','Obs_SA(T=0.5)(g)',
                  'Obs_SA(T=0.75)(g)','Obs_SA(T=1.0)(g)','Obs_SA(T=1.5)(g)',
                  'Obs_SA(T=2.0)(g)','Obs_SA(T=3.0)(g)','Obs_SA(T=4.0)(g)',
                  'Obs_SA(T=5.0)(g)','Obs_SA(T=7.5)(g)','Obs_SA(T=10.0)(g)']
obs_SA_df = pd.DataFrame(obs_SA, columns=SA_col_headers)
                
main_df = param_df.join(obs_SA_df)

    
#%% BSSA14 rock

pgm_dict = {'BSSA14_PGA(g)':bssa14_pga_rock,'BSSA14_PGA-std':bssa14_pga_std_rock,
            'BSSA14_PGV(cm/s)':bssa14_pgv_rock,'BSSA14_PGV-std':bssa14_pgv_std_rock}
pgm_df = pd.DataFrame(pgm_dict)

SA_col_headers = ['BSSA14_SA(T=0.01)(g)','BSSA14_SA(T=0.02)(g)','BSSA14_SA(T=0.03)(g)',
                  'BSSA14_SA(T=0.05)(g)','BSSA14_SA(T=0.075)(g)','BSSA14_SA(T=0.1)(g)',
                  'BSSA14_SA(T=0.15)(g)','BSSA14_SA(T=0.2)(g)','BSSA14_SA(T=0.25)(g)',
                  'BSSA14_SA(T=0.3)(g)','BSSA14_SA(T=0.4)(g)','BSSA14_SA(T=0.5)(g)',
                  'BSSA14_SA(T=0.75)(g)','BSSA14_SA(T=1.0)(g)','BSSA14_SA(T=1.5)(g)',
                  'BSSA14_SA(T=2.0)(g)','BSSA14_SA(T=3.0)(g)','BSSA14_SA(T=4.0)(g)',
                  'BSSA14_SA(T=5.0)(g)','BSSA14_SA(T=7.5)(g)',
                  'BSSA14_SA(T=10.0)(g)']

SA_std_col_headers = ['BSSA14_SA-std(T=0.01)(g)','BSSA14_SA-std(T=0.02)(g)','BSSA14_SA-std(T=0.03)(g)',
                  'BSSA14_SA-std(T=0.05)(g)','BSSA14_SA-std(T=0.075)(g)','BSSA14_SA-std(T=0.1)(g)',
                  'BSSA14_SA-std(T=0.15)(g)','BSSA14_SA-std(T=0.2)(g)','BSSA14_SA-std(T=0.25)(g)',
                  'BSSA14_SA-std(T=0.3)(g)','BSSA14_SA-std(T=0.4)(g)','BSSA14_SA-std(T=0.5)(g)',
                  'BSSA14_SA-std(T=0.75)(g)','BSSA14_SA-std(T=1.0)(g)','BSSA14_SA-std(T=1.5)(g)',
                  'BSSA14_SA-std(T=2.0)(g)','BSSA14_SA-std(T=3.0)(g)','BSSA14_SA-std(T=4.0)(g)',
                  'BSSA14_SA-std(T=5.0)(g)','BSSA14_SA-std(T=7.5)(g)',
                  'BSSA14_SA-std(T=10.0)(g)']

SA_bssa14_rock_df = pd.DataFrame(bssa14_SA_rock, columns=SA_col_headers)
SA_std_bssa14_rock_df = pd.DataFrame(bssa14_SA_std_rock, columns=SA_std_col_headers)

pgm_res_dict = {'PGA_res':bssa14_pga_rock_res,'PGV_res':bssa14_pgv_rock_res}
pgm_res_df = pd.DataFrame(pgm_res_dict)

bssa14_rock_df = main_df.join(pgm_df.join(SA_bssa14_rock_df.join(SA_std_bssa14_rock_df)))
bssa14_rock_df.to_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_BSSA14_rock.csv',index=False)


#%% BSSA14 

pgm_dict = {'BSSA14_PGA(g)':bssa14_pga,'BSSA14_PGA-std':bssa14_pga_std,
            'BSSA14_PGV(cm/s)':bssa14_pgv,'BSSA14_PGV-std':bssa14_pgv_std}
pgm_df = pd.DataFrame(pgm_dict)

SA_col_headers = ['BSSA14_SA(T=0.01)(g)','BSSA14_SA(T=0.02)(g)','BSSA14_SA(T=0.03)(g)',
                  'BSSA14_SA(T=0.05)(g)','BSSA14_SA(T=0.075)(g)','BSSA14_SA(T=0.1)(g)',
                  'BSSA14_SA(T=0.15)(g)','BSSA14_SA(T=0.2)(g)','BSSA14_SA(T=0.25)(g)',
                  'BSSA14_SA(T=0.3)(g)','BSSA14_SA(T=0.4)(g)','BSSA14_SA(T=0.5)(g)',
                  'BSSA14_SA(T=0.75)(g)','BSSA14_SA(T=1.0)(g)','BSSA14_SA(T=1.5)(g)',
                  'BSSA14_SA(T=2.0)(g)','BSSA14_SA(T=3.0)(g)','BSSA14_SA(T=4.0)(g)',
                  'BSSA14_SA(T=5.0)(g)','BSSA14_SA(T=7.5)(g)',
                  'BSSA14_SA(T=10.0)(g)']

SA_std_col_headers = ['BSSA14_SA-std(T=0.01)(g)','BSSA14_SA-std(T=0.02)(g)','BSSA14_SA-std(T=0.03)(g)',
                  'BSSA14_SA-std(T=0.05)(g)','BSSA14_SA-std(T=0.075)(g)','BSSA14_SA-std(T=0.1)(g)',
                  'BSSA14_SA-std(T=0.15)(g)','BSSA14_SA-std(T=0.2)(g)','BSSA14_SA-std(T=0.25)(g)',
                  'BSSA14_SA-std(T=0.3)(g)','BSSA14_SA-std(T=0.4)(g)','BSSA14_SA-std(T=0.5)(g)',
                  'BSSA14_SA-std(T=0.75)(g)','BSSA14_SA-std(T=1.0)(g)','BSSA14_SA-std(T=1.5)(g)',
                  'BSSA14_SA-std(T=2.0)(g)','BSSA14_SA-std(T=3.0)(g)','BSSA14_SA-std(T=4.0)(g)',
                  'BSSA14_SA-std(T=5.0)(g)','BSSA14_SA-std(T=7.5)(g)',
                  'BSSA14_SA-std(T=10.0)(g)']

SA_bssa14_df = pd.DataFrame(bssa14_SA, columns=SA_col_headers)
SA_std_bssa14_df = pd.DataFrame(bssa14_SA_std, columns=SA_std_col_headers)

bssa14_df = main_df.join(pgm_df.join(SA_bssa14_df.join(SA_std_bssa14_df)))
bssa14_df.to_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_BSSA14.csv',index=False)

#%% ASK14 rock

pgm_dict = {'ASK14_PGA(g)':ask14_pga_rock,'ASK14_PGA-std':ask14_pga_std_rock,
            'ASK14_PGV(cm/s)':ask14_pgv_rock,'ASK14_PGV-std':ask14_pgv_std_rock}
pgm_df = pd.DataFrame(pgm_dict)

SA_col_headers = ['ASK14_SA(T=0.01)(g)','ASK14_SA(T=0.02)(g)','ASK14_SA(T=0.03)(g)',
                  'ASK14_SA(T=0.05)(g)','ASK14_SA(T=0.075)(g)','ASK14_SA(T=0.1)(g)',
                  'ASK14_SA(T=0.15)(g)','ASK14_SA(T=0.2)(g)','ASK14_SA(T=0.25)(g)',
                  'ASK14_SA(T=0.3)(g)','ASK14_SA(T=0.4)(g)','ASK14_SA(T=0.5)(g)',
                  'ASK14_SA(T=0.75)(g)','ASK14_SA(T=1.0)(g)','ASK14_SA(T=1.5)(g)',
                  'ASK14_SA(T=2.0)(g)','ASK14_SA(T=3.0)(g)','ASK14_SA(T=4.0)(g)',
                  'ASK14_SA(T=5.0)(g)','ASK14_SA(T=7.5)(g)',
                  'ASK14_SA(T=10.0)(g)']

SA_std_col_headers = ['ASK14_SA-std(T=0.01)(g)','ASK14_SA-std(T=0.02)(g)','ASK14_SA-std(T=0.03)(g)',
                  'ASK14_SA-std(T=0.05)(g)','ASK14_SA-std(T=0.075)(g)','ASK14_SA-std(T=0.1)(g)',
                  'ASK14_SA-std(T=0.15)(g)','ASK14_SA-std(T=0.2)(g)','ASK14_SA-std(T=0.25)(g)',
                  'ASK14_SA-std(T=0.3)(g)','ASK14_SA-std(T=0.4)(g)','ASK14_SA-std(T=0.5)(g)',
                  'ASK14_SA-std(T=0.75)(g)','ASK14_SA-std(T=1.0)(g)','ASK14_SA-std(T=1.5)(g)',
                  'ASK14_SA-std(T=2.0)(g)','ASK14_SA-std(T=3.0)(g)','ASK14_SA-std(T=4.0)(g)',
                  'ASK14_SA-std(T=5.0)(g)','ASK14_SA-std(T=7.5)(g)',
                  'ASK14_SA-std(T=10.0)(g)']

SA_ask14_rock_df = pd.DataFrame(ask14_SA_rock, columns=SA_col_headers)
SA_std_ask14_rock_df = pd.DataFrame(ask14_SA_std_rock, columns=SA_std_col_headers)

ask14_rock_df = main_df.join(pgm_df.join(SA_ask14_rock_df.join(SA_std_ask14_rock_df)))
ask14_rock_df.to_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_ASK14_rock.csv',index=False)

#%% ASK14 

pgm_dict = {'ASK14_PGA(g)':ask14_pga,'ASK14_PGA-std':ask14_pga_std,
            'ASK14_PGV(cm/s)':ask14_pgv,'ASK14_PGV-std':ask14_pgv_std}
pgm_df = pd.DataFrame(pgm_dict)

SA_col_headers = ['ASK14_SA(T=0.01)(g)','ASK14_SA(T=0.02)(g)','ASK14_SA(T=0.03)(g)',
                  'ASK14_SA(T=0.05)(g)','ASK14_SA(T=0.075)(g)','ASK14_SA(T=0.1)(g)',
                  'ASK14_SA(T=0.15)(g)','ASK14_SA(T=0.2)(g)','ASK14_SA(T=0.25)(g)',
                  'ASK14_SA(T=0.3)(g)','ASK14_SA(T=0.4)(g)','ASK14_SA(T=0.5)(g)',
                  'ASK14_SA(T=0.75)(g)','ASK14_SA(T=1.0)(g)','ASK14_SA(T=1.5)(g)',
                  'ASK14_SA(T=2.0)(g)','ASK14_SA(T=3.0)(g)','ASK14_SA(T=4.0)(g)',
                  'ASK14_SA(T=5.0)(g)','ASK14_SA(T=7.5)(g)',
                  'ASK14_SA(T=10.0)(g)']

SA_std_col_headers = ['ASK14_SA-std(T=0.01)(g)','ASK14_SA-std(T=0.02)(g)','ASK14_SA-std(T=0.03)(g)',
                  'ASK14_SA-std(T=0.05)(g)','ASK14_SA-std(T=0.075)(g)','ASK14_SA-std(T=0.1)(g)',
                  'ASK14_SA-std(T=0.15)(g)','ASK14_SA-std(T=0.2)(g)','ASK14_SA-std(T=0.25)(g)',
                  'ASK14_SA-std(T=0.3)(g)','ASK14_SA-std(T=0.4)(g)','ASK14_SA-std(T=0.5)(g)',
                  'ASK14_SA-std(T=0.75)(g)','ASK14_SA-std(T=1.0)(g)','ASK14_SA-std(T=1.5)(g)',
                  'ASK14_SA-std(T=2.0)(g)','ASK14_SA-std(T=3.0)(g)','ASK14_SA-std(T=4.0)(g)',
                  'ASK14_SA-std(T=5.0)(g)','ASK14_SA-std(T=7.5)(g)',
                  'ASK14_SA-std(T=10.0)(g)']

SA_ask14_df = pd.DataFrame(ask14_SA, columns=SA_col_headers)
SA_std_ask14_df = pd.DataFrame(ask14_SA_std, columns=SA_std_col_headers)

ask14_df = main_df.join(pgm_df.join(SA_ask14_df.join(SA_std_ask14_df)))
ask14_df.to_csv('/Users/tnye/bayarea_path/files/residual_analysis/IM_flatfiles/IMs_ASK14.csv',index=False)


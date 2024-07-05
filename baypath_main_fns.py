#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:55:40 2024

@author: tnye
"""

###############################################################################
# This module contains functions used for this project on SFBA path effects.
###############################################################################


def bssa14(IM,M,Rrup,Rjb,ztor=7.13,rake=0.0,vs30=760,vs30_meas=False,z1pt0=1.0,period=None):
    """
    Computes PGA with Boore et al. (2014) GMPE using OpenQuake engine.
    
        Inputs:
            M(float): Magnitude
            Rrup(float): Closest rupture distance in km
            Rjb(float): Joyner-Boore rupture distance in km
            ztor(float): Depth to top of rupture. Default: 7.13 from Annemarie. ASSUMPTION IS RRUP > ZTOR
            rake(float): Rake of fault. Default is 0 degrees.
            dip(float): Dip of fault. Default is 90 degrees. 
            width(float): Width of fault. Default is 10 km.
            period(float): Period desired if estimating spectral acceleration
            
        Return:
            lmean_boore14(float): Mean PGA
            sd_boore14(float): Standard deviation of PGA
            
    """

    import numpy as np
    import pandas as pd
    from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
    from openquake.hazardlib import const, imt
    from openquake.hazardlib.contexts import RuptureContext
    from openquake.hazardlib.contexts import DistancesContext
    from openquake.hazardlib.contexts import SitesContext
    
    # Initiate model
    boore  = BooreEtAl2014()
    
    # Define intensity measure
    if IM == 'PGA':
        IMT = imt.PGA()
    elif IM == 'PGV':
        IMT = imt.PGV()
    elif IM == 'SA':
        IMT = imt.SA(period)
        
    # Initiate the rupture, distances, and sites objects:
    rctx = RuptureContext()
    dctx = DistancesContext()
    sctx = SitesContext()
    
    # Fill the rupture context...assuming rake is 0, dip is 90
    rctx.rake = rake 
    rctx.ztor = ztor
    
    # Scenario I: If M and Rrup are both single values:
    #   Then set the magnitude as a float, and the rrup/distance as an array
    #   of one value
    if isinstance(M,float) & isinstance(Rrup,float):
        rctx.mag = M
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)

        # Then compute everything else...
        #   Assuming average ztor, get rjb:
        dctx.rjb = Rjb
        # dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        
        # #   Set site parameters
        # sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        # sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sitecol_dict = {'sids':[1],'vs30':vs30,
                        'vs30measured':vs30_meas,'z1pt0':None,
                        'z2pt5':None}
        sitecollection = pd.DataFrame(sitecol_dict)
        sctx = SitesContext(sitecol=sitecollection)
        
        #  Compute prediction
        lnmean_boore14, sd_boore14 = boore.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return(lnmean_boore14, sd_boore14)
    
    
def ask14(IM,M,Rrup,Rjb,ztor=7.13,rake=0.0,dip=90.0,width=10.0,vs30=1180,vs30_meas=False,z1pt0=0.003,period=None):
    """
    Computes PGA or PGV with Abrahamson et al. (2014) GMPE using OpenQuake engine.
    
        Inputs:
            imt(string): IM (PGA, PGV, or SA)
            M(float): Magnitude
            Rrup(float): Closest rupture distance in km
            Rjb(float): Joyner-Boore rupture distance in km
            vs30(float): Vs30 in m/s
            vs30_meas(float): Measured Vs30 in m/s
            ztor(float): Depth to top of rupture. Default: 7.13 from Annemarie. ASSUMPTION IS RRUP > ZTOR
            rake(float): Rake of fault. Default is 0 degrees.
            dip(float): Dip of fault. Default is 90 degrees. 
            width(float): Width of fault. Default is 10 km.
            z1pt0(float): Soil depth to Vs = 1km/s, in m.  Default is 50.
            period(float): Period desired if estimating spectral acceleration
            
        Return:
            lmean_ask14(float): ln Mean PGA in units of %g
            sd_ask14(float): Standard deviation of PGA
            
    """

    import numpy as np
    import pandas as pd
    from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014
    from openquake.hazardlib import const, imt
    from openquake.hazardlib.contexts import RuptureContext
    from openquake.hazardlib.contexts import DistancesContext
    from openquake.hazardlib.contexts import SitesContext
    
    # Initiate model
    ASK14  = AbrahamsonEtAl2014()
    
    # Define intensity measure
    if IM == 'PGA':
        IMT = imt.PGA()
    elif IM == 'PGV':
        IMT = imt.PGV()
    elif IM == 'SA':
        IMT = imt.SA(period)
    
    # Initiate the rupture, distances, and sites objects:
    rctx = RuptureContext()
    dctx = DistancesContext()
    sctx = SitesContext()
    
    # Fill the rupture context...assuming rake is 0, dip is 90
    rctx.rake = rake
    rctx.dip = dip
    rctx.ztor = ztor
    rctx.width = width 
    rctx.mag = M
    dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)

    # Then compute everything else...
    #   Assuming average ztor, get rjb:
    dctx.rjb = Rjb
    # dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
    dctx.rhypo = dctx.rrup
    dctx.rx = dctx.rjb
    dctx.ry0 = dctx.rx
    
    # Set site parameters
    sitecol_dict = {'sids':[1],'vs30':vs30,
                    'vs30measured':vs30_meas,'z1pt0':z1pt0,
                    'z2pt5':None}
    sitecollection = pd.DataFrame(sitecol_dict)
    sctx = SitesContext(sitecol=sitecollection)
    
    #  Compute prediction
    lnmean_ask14, sd_ask14 = ASK14.get_mean_and_stddevs(sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
        
    return(lnmean_ask14, sd_ask14)



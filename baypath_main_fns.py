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
                    'vs30measured':vs30_meas,'z1pt0':z1pt0,
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


def lonlat_to_utm(lon, lat):
    """
    Converts lon/lat coordinates to UTM coordinates.
    
        Inputs:
            lon(float): longitude of datapoint
            lat(float): latitude of datapoint
        
        Return:
            utmx(float): UTMx coordiante of datapoint
            utmy(float): UTMy coordinate of datapoint
    """
    
    from pyproj import Proj
    
    # Convert datapoint from lonlat to utm while in Domain A
    proj = Proj(proj='utm', zone=10, datum='WGS84')
    utmx, utmy = proj(lon, lat)
    
    return(utmx, utmy)


def domainA_to_domainC(lonA, latA):
    """
    Converts lon/lat coordinates in Domain A (true lon/lat)
    to UTM coordinates in Domain C (rotated to standard x-y reference frame with
    bottom left corner of velocity model at 0,0)
    
        Inputs:
            lonA(float): longitude of datapoint
            latA(float): latitude of datapoint
            
        Return:
            utmxC(float): X coordinate of datapoint in Domain C
            utmyC(float): Y coordinate of datapoint in Domain C
            
    """
    
    import numpy as np
    from pyproj import Proj
    
    # Angle to rotate velocity model to for standard x-y reference system
    rot_angle = 36.36237154586088
    angle_rad = np.deg2rad(rot_angle)
    
    # Velocity model bounding coords in Domain A UTM
    cornersA_utmx = [599289.0988159698, 430310.85993747227, 539026.9666502575, 708001.3539085807]
    cornersA_utmy = [4023027.3900981518, 4252529.623456127, 4332565.171409805, 4103066.8269178835]
    
    # Convert datapoint from lonlat to utm while in Domain A
    proj = Proj(proj='utm', zone=10, datum='WGS84')
    utmxA, utmyA = proj(lonA, latA)

    # Translate coordinates to Domain B (bottom left corner of velocity model at (0,0))
    utmxB = np.array(utmxA) - cornersA_utmx[0]
    utmyB = np.array(utmyA) - cornersA_utmy[0]

    # Rotation matrix to rotate coordinate around the origin to get into a
        # standard x-y reference frame
    rotation_matrix = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                                [-np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Perform rotation 
    utmxC, utmyC = np.dot(rotation_matrix, np.array([utmxB, utmyB]))
    
    return(utmxC, utmyC)


def domainC_to_domainA(utmxC, utmyC):
    """
    Converts UTM coordinates in Domain C (pykonal reference frame)
    to UTM coordinates in Domain A (true UTMx/UTMy)
    
        Inputs:
            utmxC(float): X coordinate of datapoint in Domain C
            utmyC(float): Y coordinate of datapoint in Domain C
            
        Return:
            utmxA(float): X coordinate of datapoint in Domain A
            utmyA(float): Y coordinate of datapoint in Domain A
            
    """
    
    import numpy as np
    
    # Angle to rotate velocity model to for standard x-y reference system
    rot_angle = -36.36237154586088
    angle_rad = np.deg2rad(rot_angle)
    
    # Velocity model bounding coords in Domain A UTM
    cornersA_utmx = [599289.0988159698, 430310.85993747227, 539026.9666502575, 708001.3539085807]
    cornersA_utmy = [4023027.3900981518, 4252529.623456127, 4332565.171409805, 4103066.8269178835]
    
    # Rotation matrix to rotate coordinate around the origin to get into 
        # velocity model reference frame
    rotation_matrix = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                                [-np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Perform rotation 
    utmxB, utmyB = np.dot(rotation_matrix, np.array([utmxC, utmyC]))
    
    # Translate coordinates back to the original velocity model reference frame
    utmxA = np.array(utmxB) + cornersA_utmx[0]
    utmyA = np.array(utmyB) + cornersA_utmy[0]
    
    return(utmxA, utmyA)



def interpolate_ray(ray_x, ray_y, ray_z, spacing=10):
    """
    Interpolates raypath given a set spacing interval.
    
        Inputs:
            ray_x(float): X-coordinates of raypath (m)
            ray_y(float): Y-coordinates of raypath (m)
            ray_z(float): Z-coordinates of raypath (m)
            spacing(float): Desired raypath coordinate spacing (m) 
            
        Return:
            new_x(float): Interpolated X-coordinates (m)
            new_y(float): Interpolated Y-coordinates (m)
            new_z(float): Interpolated Z-coordinates (m)
            
    """
    
    import numpy as np
    from scipy.interpolate import interp1d
    
    # Compute the distances between consecutive points
    dx = np.diff(ray_x)
    dy = np.diff(ray_y)
    dz = np.diff(ray_z)
    distances = np.sqrt(dx**2 + dy**2 + dz**2)

    # Cumulative arc length
    arc_length = np.concatenate([[0], np.cumsum(distances)])

    # Define the new arc length values with desired spacing
    new_arc_length = np.arange(0, arc_length[-1], spacing)

    # Interpolate the XYZ coordinates
    interp_x = interp1d(arc_length, ray_x, kind='quadratic')
    interp_y = interp1d(arc_length, ray_y, kind='quadratic')
    interp_z = interp1d(arc_length, ray_z, kind='quadratic')

    # Resample the raypath
    new_x = interp_x(new_arc_length)
    new_y = interp_y(new_arc_length)
    new_z = interp_z(new_arc_length)

    return(new_x, new_y, new_z)


def compute_path_integral(ray_val, max_material_val):
    """
    Computes the path integral of a model medium (e.g., velocity or Q).
    From Sahakian et al. (2019)
    
        Inputs:
            ray_val(array): Array of model medium values along a raypath.
            max_material_val(float): Maximum material value in the model.
            
        Return:
            path_integral(float): Normalized path integral. 
            
    """
    
    import numpy as np
    
    path_integral = np.sum(sum(abs(ray_val)))
    
    #Get the length of this ray:
    ray_len=len(ray_val)
    
    max_val_integral=ray_len*max_material_val
    
    #Sum the values of this ray:
    path_integral=sum(ray_val)
    
    #Set the value of the index for this ray:
    norm_path_integral=path_integral/max_val_integral
        
    
    return(norm_path_integral)
    

def compute_devpathintegral(ray_val):
    """
    Computes the gradient of a model medium path integral (e.g., velocity or Q).
    From Sahakian et al. (2019)
    
        Inputs:
            ray_val(array): Array of model medium values along a raypath.
            
        Return:
            dpath_integral(float): Gradient path integral .
            
    """
    
    import numpy as np
    
    # Compute velocity gradient
    gradient = np.gradient(ray_val)
    
    dpath_integral = np.sum(abs(gradient))
    
    return(dpath_integral)

def compute_devpathintegral_surface(ray_data):
    """
    Computes the gradient of a model medium path integral (e.g., velocity or Q)
    for a surface wave, considering a column of subsurface structure.
    From Sahakian et al. (2019)
    
        Inputs:
            ray_val(array): Array of model medium values along a raypath.
            
        Return:
            dpath_integral(float): Gradient path integral .
            
    """
    
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.integrate import simps
    from scipy.integrate import trapezoid
    
    def interpolate_velocity(x, y, depth):
        # Find nearest surface point
        idx = np.argmin(np.abs(x_surface - x) + np.abs(y_surface - y))
        depth_values = depths
        velocity_values = velocities_depth[idx, :]
        interp_func = interp1d(depth_values, velocity_values, fill_value="extrapolate")
        return interp_func(depth)
    
    def integrate_velocity_along_depth(x, y):
        depth_values = np.array([0, 1, 2])  # Example depth values
        velocities = np.array([interpolate_velocity(x, y, d) for d in depth_values])
        return simps(velocities, depth_values)  # Simpson's rule for numerical integration
    
    x_surface = ray_data[:,0][np.where(ray_data[:,2] == 0)[0]]
    y_surface = ray_data[:,1][np.where(ray_data[:,2] == 0)[0]]
    depths = np.unique(ray_data[:,2])

    # Compute the integral at each surface point
    depth_integrals = np.array([integrate_velocity_along_depth(x, y) for x, y in zip(x_surface, y_surface)])
    
    # Integrate over the surface path
    def integrate_along_path(x_surface, y_surface, depth_integrals):
        return trapezoid(depth_integrals, x=x_surface)  # or y_surface if using y for the path
    
    # Compute the final integral
    total_integral = integrate_along_path(x_surface, y_surface, depth_integrals)
    
    # Compute velocity gradient
    gradient = np.gradient(ray_val)
    
    dpath_integral = np.sum(abs(gradient))
    
    return(dpath_integral)


# def interpolate_rays(ray_coords,materialobj,interptype,modeltype='grid'):
#     '''
#     Interpolate rays through a material model
#     Input:
#         residobj:           Object with residuals and ray position information
#         materialobj:        Object with material model.  Depth should be positive for input.
#         interptype:         String with type of interpolation from rbf, i.e., 'linear'
#         modeltype:          String with model type.  Default: 'grid' (m x n x p array that is a material model class). 
#                                 or: 'nodes', an n x 4 array, where the 4 columns are lon, lat, depth, Q and n is the number of nodes.
#     Output:
#         ray_data:           List of arrays, each of same length as corresponding ray.
#                             Arrays contain values of model at those raypath positions.
#     '''
    
#     from numpy import meshgrid,zeros,where
#     from scipy.interpolate import Rbf

#     # If the model type is a grid:
#     if modeltype == 'grid':
        
#         # #Get the actual grid x and y values:
#         # grid_x=materialobj.X.values
#         # grid_y=materialobj.Y.values
#         # grid_z=-materialobj['Depth(m)'].values
    
#         # #Turn these into a grid, prepping for ravel for interpolation:
#         # gX,gY,gZ=meshgrid(grid_x,grid_y,grid_z,indexing='ij')
    
#         #Make column vectors to put into the interpolation:
#         columnx=materialobj.X.values
#         columny=materialobj.Y.values
#         columnz=materialobj['Depth(m)'].values
#         #Data to interpolate - transpose so the x is first, etc.:
#         data=materialobj.materials.transpose(2,1,0).ravel()  
    
#     elif modeltype == 'nodes':
#         columnx = materialobj.X.values
#         columny = materialobj.Y.values
#         columnz = materialobj['Depth(m)'].values
#         data = materialobj['Vs(m/s)'].values
    
#     #Make interpolator
#     interpolator = Rbf(columnx, columny, columnz, data,function=interptype)

#     #Get the values to interpolate., then interpolate
#     ray_x_i=residobj.vs_lon[ray_i]
#     ray_y_i=residobj.vs_lat[ray_i]
#     ray_z_i=residobj.vs_depth[ray_i]
        
#     #Interpolate:
#     vs_i=interpolator(ray_data[:,0],ray_data[:,1],ray_data[:,2])

#     #Append to the list:
#     ray_data.append(vs_i)            
    
#     #Print position:
#     if ray_i % 1000 ==0: print ray_i
            
#     #Return info:
#     return ray_data


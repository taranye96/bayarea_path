#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 13:30:38 2024

@author: tnye
"""

import numpy as np
import pandas as pd
from pyproj import Proj, transform
import matplotlib.pyplot as plt

# Function to convert lon/lat to UTM
def lonlat_to_utm(lon, lat):
    proj = Proj(proj='utm', zone=10, datum='WGS84')
    utm_x, utm_y = proj(lon, lat)
    return utm_x, utm_y

# Function to convert UTM to lon/lat
def utm_to_lonlat(utm_x, utm_y):
    proj = Proj(proj='utm', zone=10, datum='WGS84')
    lon, lat = proj(utm_x, utm_y, inverse=True)
    return lon, lat

# Function to rotate points
def rotate_point(x, y, angle_deg):
    
    angle_rad = np.deg2rad(angle_deg)

    rotation_matrix = np.array([
    [np.cos(angle_rad), np.sin(angle_rad)],
    [-np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    rot_x, rot_y = np.dot(rotation_matrix, np.array([x, y]))
    
    return rot_x, rot_y

#%% Rotate coordinates to true UTMx-UTMy reference frame

# Define the rotation angle in degrees
rot_angle = 36.632

# Velocity model bounding coords
orig_lon = [-121.8935, -123.7983, -122.5484, -120.6609, ]
orig_lat = [36.3472, 38.4183, 39.1414, 37.0508]

# Convert lonlat coordinates to UTM
orig_utm_x, orig_utm_y = lonlat_to_utm(orig_lon, orig_lat)

# Translate coordinates to origin
trans_utm_x = np.array(orig_utm_x) - orig_utm_x[0]
trans_utm_y = np.array(orig_utm_y) - orig_utm_y[0]

# Rotate the UTM coordinates around the origin
rot_utm_x, rot_utm_y = rotate_point(trans_utm_x, trans_utm_y, rot_angle)


#%% Define model nodes

# Create the individual depth arrays with different spacing
z250 = np.arange(-21000, -10000, 250)
z125 = np.arange(-10000, -3500, 125)
z50 = np.arange(-3500, -500, 50)
z25 = np.arange(-500, 25, 25)

# Concatenate the depth arrays
z_coords = np.concatenate((z250, z125, z50, z25))

# Initialize arrays for the rotated grid of points
X_rot = np.array([])
Y_rot = np.array([])
Z = np.array([])

# Get min and max bounds of grid
utm_x_min, utm_x_max, utm_y_min, utm_y_max = 0, np.max(rot_utm_x), 0, np.max(rot_utm_y)

# Loop over depth arrays and determine x and y coordinate spacing
for z in z_coords:
    if z < -10000:
        x = np.arange(utm_x_min, utm_x_max+1, 1000)
        y = np.arange(utm_y_min, utm_y_max+1, 1000)
    elif z >= -10000 and z < -3500:
        x = np.arange(utm_x_min, utm_x_max+1, 500)
        y = np.arange(utm_y_min, utm_y_max+1, 500)
    elif z >= -3500 and z < -500:
        x = np.arange(utm_x_min, utm_x_max+1, 200)
        y = np.arange(utm_y_min, utm_y_max+1, 200)
    elif z >= -500:
        x = np.arange(utm_x_min, utm_x_max+1, 100)
        y = np.arange(utm_y_min, utm_y_max+1, 100)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    X_rot = np.append(X_rot, xx)
    Y_rot = np.append(Y_rot, yy)
    Z = np.append(Z, zz)


#%% Rotate coordinates back to the original reference frame

# Unrotate the grid of nodes
trans_X, trans_Y = rotate_point(X_rot, Y_rot, -1*rot_angle)

# Translate coordinates from origin back to original locaation
X = np.array(trans_X) + orig_utm_x[0]
Y = np.array(trans_Y) + orig_utm_y[0]

# Convert UTM coordinates to lonlat
lon_grid, lat_grid = utm_to_lonlat(X, Y)


#%% Figures to make sure coordinates look right

# Does the map look right?
plt.figure(figsize=(10, 10))
# plt.plot(lon[np.where(Z == 0)[0]], lat[np.where(Z == 0)[0]])
plt.scatter(lon_grid[np.where(Z == 0)[0]], lat_grid[np.where(Z == 0)[0]])
plt.scatter(orig_lon, orig_lat)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Does the map look right?
plt.figure(figsize=(10, 10))
# plt.plot(X_rot[np.where(Z == 0)[0]], Y_rot[np.where(Z == 0)[0]])
# plt.plot(X[np.where(Z == 0)[0]], Y[np.where(Z == 0)[0]])
# plt.plot(trans_utm_x, trans_utm_y)
# plt.plot(rot_utm_x, rot_utm_y)
plt.scatter(X[np.where(Z == 0)[0]], Y[np.where(Z == 0)[0]])
plt.scatter(orig_utm_x, orig_utm_y)
plt.xlabel('UTMx')
plt.ylabel('UTMy')
plt.grid(True)
plt.show()


#%% Format data for geomodel input files

# Combine X, Y, Z into a single array for sorting
coordinates = np.column_stack((X, Y, Z))

# Sort coordinates primarily by X, then by Y, then by Z
sorted_indices = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
sorted_coordinates = coordinates[sorted_indices]

# Extract sorted X, Y, Z coordinates
sorted_X = sorted_coordinates[:, 0].reshape(X.shape)
sorted_Y = sorted_coordinates[:, 1].reshape(Y.shape)
sorted_Z = sorted_coordinates[:, 2].reshape(Z.shape)

utm_points = np.stack([sorted_X, sorted_Y, sorted_Z], axis=-1)
np.savetxt('/Users/tnye/bayarea_path/files/velmod/USGS_SFCVM_v21-1_detailed_utm.in', utm_points, delimiter='\t')


# Does the map look right?
plt.figure(figsize=(10, 10))
plt.scatter(sorted_X[np.where(sorted_Z == 0)[0]], sorted_Y[np.where(sorted_Z == 0)[0]],marker='.')
plt.scatter(orig_utm_x, orig_utm_y)
plt.xlabel('UTMx')
plt.ylabel('UTMy')
plt.grid(True)
plt.show()
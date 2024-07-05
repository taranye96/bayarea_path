#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:17:45 2024

@author: tnye
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:03:04 2024

@author: tnye
"""

import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt

# Define the latitude and longitude bounds
lat_min, lat_max = 36, 39
lon_min, lon_max = -123.5, -121

# Set up the projection to UTM zone 10N (suitable for this region)
proj = pyproj.Proj(proj='utm', zone=10, ellps='WGS84')

# Convert the bounding box corners to UTM
utm_min_x, utm_min_y = proj(lon_min, lat_min)
utm_max_x, utm_max_y = proj(lon_max, lat_max)

# Create the individual arrays with different spacing
z250 = np.arange(-21000, -10000, 250)
z125 = np.arange(-10000, -3500, 125)
z50 = np.arange(-3500, -500, 50)
z25 = np.arange(-500, 25, 25)

# Concatenate the arrays
z_coords = np.concatenate((z250, z125, z50, z25))

X = np.array([])
Y = np.array([])
Z = np.array([])

for z in z_coords:
    if z < -10000:
        x = np.arange(utm_min_x, utm_max_x+1, 1000)
        y = np.arange(utm_min_y, utm_max_y+1, 1000)
    elif z >= -10000 and z < -3500:
        x = np.arange(utm_min_x, utm_max_x+1, 500)
        y = np.arange(utm_min_y, utm_max_y+1, 500)
    elif z >= -3500 and z < -500:
        x = np.arange(utm_min_x, utm_max_x+1, 200)
        y = np.arange(utm_min_y, utm_max_y+1, 200)
    elif z >= -500:
        x = np.arange(utm_min_x, utm_max_x+1, 100)
        y = np.arange(utm_min_y, utm_max_y+1, 100)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    X = np.append(X, xx)
    Y = np.append(Y, yy)
    Z = np.append(Z, zz)
    
# Create a grid of points within the UTM bounding box
x_coords = np.unique(X.ravel())
y_coords = np.unique(Y.ravel())

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
np.savetxt('/Users/tnye/bayarea_path/files/velmod/bay_3Dvelmod_utm-cartesian.in', utm_points, delimiter='\t')


# # Plot the raypath
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(sorted_X[np.where(sorted_Y == utm_min_y)[0]], sorted_Z[np.where(sorted_Y == utm_min_y)[0]], color='k', marker='.', s=1)
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Z (m)')



#%%



utm_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

# Separate UTM coordinates and depth values
utm_x = utm_points[:, 0]
utm_y = utm_points[:, 1]
depth = utm_points[:, 2]

# Set up the projection to UTM zone 10N (suitable for this region)
proj = pyproj.Proj(proj='utm', zone=10, ellps='WGS84')

# Convert UTM coordinates to latitude and longitude
lon, lat = proj(utm_x, utm_y, inverse=True)

# Combine latitude, longitude, and depth into a new array
latlon_points = np.column_stack((lat, lon, depth))

# Does the map look right?
plt.figure(figsize=(10, 10))
plt.plot(latlon_points[:, 1], latlon_points[:, 0], 'b.', markersize=2)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Does the map look right?
plt.figure(figsize=(10, 10))
plt.plot(utm_points[:, 0], utm_points[:, 1], 'b.', markersize=2)
plt.xlabel('UTMx')
plt.ylabel('UTMy')
plt.grid(True)
plt.show()
 
# Save file
np.savetxt('/Users/tnye/bayarea_path/files/velmod/bay_3Dvelmod_utm-cartesian.in', utm_points, delimiter='\t')

#%%

# # Initialize an empty DataFrame
# df_all_depths = pd.DataFrame(columns=['Latitude', 'Longitude', 'Depth(m)'])

# # Loop through each depth and append the coordinates with depth to the DataFrame
# for depth in depths:
#     df = pd.DataFrame(lonlat_points, columns=['Longitude', 'Latitude'])
#     df['Depth(m)'] = depth
#     df_all_depths = pd.concat([df_all_depths, df], ignore_index=True)

# Save file
np.savetxt('/Users/tnye/bayarea_path/files/velmod/bay_3Dvelmod_lonlat-cartesian.in', latlon_points, delimiter='\t')

xx, yy=np.meshgrid(x_coords, y_coords)
X_flat = xx.ravel()
Y_flat = yy.ravel()
# Convert UTM coordinates to latitude and longitude
GE_lon, GE_lat = proj(X_flat, Y_flat, inverse=True)

# Create DataFrame using the flattened arrays
df = pd.DataFrame({'Lon': GE_lon, 'Lat': GE_lat})

df.to_csv('/Users/tnye/bayarea_path/files/velmod/bay_3Dvelmod_lonlat-cartesian.csv')


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
lat_min, lat_max = 35.9, 39
lon_min, lon_max = -123.5, -121

# Define the rotation angle
rotation_angle_deg = 36.362
rotation_angle_rad = np.radians(rotation_angle_deg)

# Set up the projection to UTM zone 10N (suitable for this region)
proj = pyproj.Proj(proj='utm', zone=10, ellps='WGS84')

# Convert the bounding box corners to UTM
utm_min_x, utm_min_y = proj(lon_min, lat_min)
utm_max_x, utm_max_y = proj(lon_max, lat_max)

# Determine the rotated extents of the bounding box
bbox_corners = np.array([
    [utm_min_x, utm_min_y],
    [utm_max_x, utm_min_y],
    [utm_max_x, utm_max_y],
    [utm_min_x, utm_max_y]
])
center = np.mean(bbox_corners, axis=0)
rot_matrix = np.array([
    [np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
    [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)]
])
rotated_corners = np.dot(bbox_corners - center, rot_matrix.T) + center
rotated_min_x, rotated_min_y = rotated_corners.min(axis=0)
rotated_max_x, rotated_max_y = rotated_corners.max(axis=0)

# Define grid spacing in meters
grid_spacing = 1000

# Create a grid of points within the rotated bounding box
x_coords = np.arange(rotated_min_x, rotated_max_x, grid_spacing)
y_coords = np.arange(rotated_min_y, rotated_max_y, grid_spacing)
X, Y = np.meshgrid(x_coords, y_coords)

# Rotate the grid points back to the original UTM coordinates
rotated_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
inv_rot_matrix = np.array([
    [np.cos(-rotation_angle_rad), -np.sin(-rotation_angle_rad)],
    [np.sin(-rotation_angle_rad), np.cos(-rotation_angle_rad)]
])
center = np.array([np.mean([rotated_min_x, rotated_max_x]), np.mean([rotated_min_y, rotated_max_y])])
utm_points = np.dot(rotated_points - center, inv_rot_matrix.T) + center

# Filter the points to keep only those within the original bounding box
lon, lat = proj(utm_points[:, 0], utm_points[:, 1], inverse=True)
filtered_points = utm_points[(36 <= lat) & (lat <= lat_max) & (lon_min <= lon) & (lon <= lon_max)]
filtered_lon_lat = np.array([proj(p[0], p[1], inverse=True) for p in filtered_points])

# Depths to include
depths = [0] + list(np.arange(-250, -35001, -250))

# Initialize an empty DataFrame
df_all_depths = pd.DataFrame(columns=['Latitude', 'Longitude', 'Depth(m)'])

# Loop through each depth and append the coordinates with depth to the DataFrame
for depth in depths:
    df = pd.DataFrame(filtered_lon_lat, columns=['Longitude', 'Latitude'])
    df['Depth(m)'] = depth
    df_all_depths = pd.concat([df_all_depths, df], ignore_index=True)

# np.savetxt('/Users/tnye/bayarea_path/files/velmod/bay_3Dvelmod_utm-coords.in', filtered_points, delimiter='\t')
np.savetxt('/Users/tnye/bayarea_path/files/velmod/bay_3Dvelmod_lonlat.in', df_all_depths, delimiter='\t')



plt.figure(figsize=(10, 10))
plt.plot(filtered_lon_lat[:, 0], filtered_lon_lat[:, 1], 'b.', markersize=2)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Rotated Grid Points within Original Bounding Box (Lat/Lon)')
plt.grid(True)
plt.show()
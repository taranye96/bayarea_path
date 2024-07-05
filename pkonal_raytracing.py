#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:32:10 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import pykonal
import pyproj
import matplotlib.pyplot as plt

# Read in velocity values
velmod_file = '/Users/tnye/bayarea_path/files/velmod/USGS_SFCVM_v21-1_detailed_utm.in.out'

# Define the size of each chunk and the columns to read
chunk_size = 10**6  # Adjust the chunk size as needed

# Define the expected columns
columns = ['x0', 'x1', 'x2', 'density', 'Vp', 'Vs', 'Qp', 'Qs', 'fault_block_id', 'zone_id']

# Initialize an empty list to hold data chunks
chunks = []

# Read the file in chunks, skipping the initial lines and setting the correct column names
for chunk in pd.read_csv(velmod_file, 
                         chunksize=chunk_size, 
                         delimiter=r'\s+', 
                         comment='#',
                         skiprows=2, 
                         header=None):
    chunk.columns = columns  # Set the column names
    chunks.append(chunk[['x0', 'x1', 'x2', 'Vp', 'Vs']])  # Select the required columns

# Concatenate all chunks into a single DataFrame
velmod = pd.concat(chunks, axis=0)


# Extract coordinates and velocity values
utm_x = velmod['x0'].values
utm_y = velmod['x1'].values
depth = velmod['x2'].values
Vp = velmod['Vp'].values
Vs = velmod['Vs'].values

node_coords = np.column_stack((utm_x, utm_y, depth))

# Find the unique coordinates to determine grid dimensions
x_coords = np.unique(utm_x)
y_coords = np.unique(utm_y)
z_coords = np.unique(depth)

solver_nodes = []
V = []
for x in x_coords:
    
    tmp_x = []
    tmp_xv = []
    x_idx = np.where(utm_x == x)[0]
    y_coords = np.unique(utm_y[x_idx])
    
    for y in y_coords:
        
        tmp_y = []
        tmp_yv = []
        xy_idx = np.where((utm_x == x) & (utm_y == y))[0]
        z_coords = np.unique(depth[xy_idx])
        
        for z in z_coords:
            xyz_idx = np.where((utm_x == x) & (utm_y == y) & (depth == z))[0][0] 
            tmp_y.append([x, y, z])
            tmp_yv.append(Vp[xyz_idx])
            
        tmp_x.append(np.array(tmp_y))
        tmp_xv.append(np.array(tmp_yv))
    solver_nodes.append(np.array(tmp_x))
    V.append(np.array(tmp_xv))
solver_nodes = np.array(solver_nodes)   
            

# Define grid parameters
nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
# dx, dy, dz = 1000, 1000, 250  # Node intervals in meters


# utm_points = []
# for x in range(len(x_coords)):
#     tmp_x = []
#     for y in range(len(y_coords)):
#         tmp_y = []
#         for z in range(len(z_coords)):
#             tmp_y.append([x_coords[x], y_coords[y], z_coords[z]])
#         tmp_x.append(np.array(tmp_y))
#     utm_points.append(np.array(tmp_x))
# utm_points = np.array(utm_points)

# Create an empty velocity array
velocity_grid = np.full((nx, ny, nz), np.nan)

# Fill the velocity array with values from the data
for i in range(len(utm_x)):
    ix = np.where(x_coords == utm_x[i])[0][0]
    iy = np.where(y_coords == utm_y[i])[0][0]
    iz = np.where(z_coords == depth[i])[0][0]
    velocity_grid[ix, iy, iz] = Vp[i]
    
    
# # Create an interpolator
# interpolator = RegularGridInterpolator((utm_x, utm_y, depth), Vp, method='linear')

# # Extract velocities along the ray path
# velocities_along_ray = interpolator(ray)



#%%


# Create a PyKonal solver instance for Cartesian coordinates
solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")

solver.velocity.min_coords = [np.min(utm_x), np.min(utm_y), np.min(depth)]  # Adjust minimum coordinates as per your grid
solver.velocity.node_intervals = [None, None, None]  # Not used for non-uniform grids
solver.velocity.npts = [len(x_coords), len(y_coords), len(z_coords)]  # Number of nodes in x, y, z

node_coords = np.array([])
for x in x_coords:
    tmp_x = []
    for y in y_coords:
        tmp_y = []
        for z in z_coords:
            try:
                idx = np.where((utm_x==x) & (utm_y==y) & (depth==z))[0][0]
                tmp_y.append([x, y, z])
            except:
                continue
            
        tmp_x.append(np.array(tmp_y))
    node_coords.append(np.array(tmp_x))
node_coords = np.array(node_coords)
        
    if == 0:
        idx = np.where
solver.velocity.nodes = node_coords.reshape(len(x_coords), len(y_coords), len(z_coords), 3)
solver.velocity.nodes = solver_nodes
solver.velocity.values = V

# Define the computational domain in Cartesian coordinates
solver.vv.min_coords = [x_coords.min(), y_coords.min(), z_coords.min()]  # Minimum coordinates in x, y, z
solver.velocity.node_intervals = [None, None, None]  # Not used for non-uniform grids
solver.velocity.npts = [len(x_coords), len(y_coords), len(z_coords)]  # Number of nodes in x, y, z

solver.vv.node_intervals = [dx, dy, dz]
solver.vv.npts = [nx, ny, nz]  # Number of nodes in x, y, z

# Example velocity model (uniform velocity for demonstration)
solver.vv.values = velocity_grid  # Assign velocity values to the solver

# Define the source location in Cartesian coordinates
# src_loc = np.array([5.549360e+05,  4.084064e+06, -1.500000e+03], dtype=np.float64)
# src_loc = np.array([513936.0, 3984064.0, -1000], dtype=np.float64)
src_loc = np.array([utm_x[0], utm_y[0], depth[0]], dtype=np.float64)

# Define the receiver location at the surface

# Set the source location for the solver
solver.src_loc = src_loc

# Solve the Eikonal equation for the upgoing ray to the receiver location
print("Solving the Eikonal equation for the upgoing ray...")
solver.solve()

# Plot the ray path
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
ax.scatter(utm_x, depth, c=solver.tt.values.ravel(), cmap='plasma_r', vmin=0, vmax=60, marker='.', alpha=0.1)
ax.scatter(src_loc[0], src_loc[2], marker='*', c='goldenrod', s=200)
mappable = plt.cm.ScalarMappable(cmap='plasma_r')
mappable.set_array(solver.tt.values.ravel())
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Travel time (s)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Depth (m)')

# Define the receiver location at the surface
# rx_loc = np.array([6.049360e+05, 4.124064e+06, 0], dtype=np.float64)
rx_loc = np.array([613936.0, 3984064.0, 0], dtype=np.float64)

# Trace the ray from the receiver to the source
print("Tracing the ray from the receiver to the source...")
ray = solver.trace_ray(rx_loc)

# Plot the raypath
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color='k', linewidth=0.5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Ray Path')
plt.show()



# Check for NaN or Inf values in the velocity data
if np.isnan(Vs).any() or np.isinf(Vs).any():
    print("Warning: Velocity data contains NaN or Inf values. Please clean the data.")
    Vs = np.nan_to_num(Vs, nan=1.0, posinf=1.0, neginf=1.0)  # Replace NaN/Inf with a valid value


#%%

iasp91 = pd.read_csv(
    '/Users/tnye/Downloads/IASP91.csv',
    header=None,
    names=["depth", "radius", "Vp", "Vs"]
)

# Define how many nodes to use in the radial and azimuthal directions.
# nrho, nphi = 256, 256
nrho, nphi = 256, 256

# An empty list to hold solvers for different segments of the propagation.
solvers = []

# Define the source location.
src_loc = 5971, np.pi/2, np.pi/2

# This parameter defines segments of the propagation path. The wavefronts go
# from the source to the inner core boundary, then from the inner-core boundary
# to the Earth's surface.
EARTH_RADIUS = 6371 # Earth radius in km.
CMB = 3480 # Core-mantle boundary radius in km.
ICB = 1220 # Inner-core outer-core boundary radius in km.
path = (
    (EARTH_RADIUS, ICB), # This defines the first segment of the path.
    (ICB, EARTH_RADIUS) # This defines the second segment.
    )

# This parameter defines segments of the propagation path. The wavefronts go
# from the source to the inner core boundary, then from the inner-core boundary
# to the Earth's surface.
path = (
    (EARTH_RADIUS, ICB), # This defines the first segment of the path.
    (ICB, EARTH_RADIUS) # This defines the second segment.
)

plt.close("all")

for ipath in range(len(path)):
    path_seg = path[ipath]

    if ipath == 0:
        # If this is the first segment of the propagation path, then use a
        # PointSourceSolver.
        solver = pykonal.solver.PointSourceSolver(coord_sys="spherical") 
    else:
        # Otherwise use an ordinary EikonalSolver.
        solver = pykonal.EikonalSolver(coord_sys="spherical")

    solvers.append(solver)

    # Define the computational domain.
    solver.vv.min_coords = min(path_seg), np.pi / 2, 0 
    solver.vv.node_intervals = (
        (max(path_seg) - min(path_seg)) / (nrho - 1), 
        1, 
        np.pi / (nphi - 1)
    )
    solver.vv.npts = nrho, 1, nphi

    # Interpolate IASP91 onto the computational grid.
    solver.vv.values = np.interp(
        solver.vv.nodes[..., 0],
        iasp91["radius"].values[-1::-1],
        iasp91["Vp"].values[-1::-1]
    )

    if ipath == 0:
        # If this is the first segment of the propagation path, set the source
        # location.
        solver.src_loc = src_loc
    else:
        # Otherwise interpolate the traveltime field of the previous segment of
        # of the propagation path onto the boundary of the current segment.
        if path_seg[0] < path_seg[1]:
            # If this is a upgoing wavefront, then interpolate onto the lower
            # boundary.
            irho = 0
        else:
            # Otherwise interpolate onto the upper boundary.
            irho = nrho - 1
            
        # Set the traveltime at each node along the boundary of the current
        # segment of the propagation path equal to the value at that position
        # from the previous segment.
        
        for iphi in range(nphi):
            idx = (irho, 0, iphi)
            node = solver.tt.nodes[idx]
            solver.tt.values[idx] = solvers[ipath-1].tt.resample(node.reshape(1, 3))
            solver.unknown[idx] = False
            solver.trial.push(*idx)

    # Finally, solve the eikonal equation for the traveltime field.
    print("Solving the eikonal equation...")
    solver.solve()
    

def plot_field(field, ax, irho=slice(None), itheta=slice(None), iphi=slice(None), cmap=plt.get_cmap("hot"), boundaries=[ICB, CMB, EARTH_RADIUS], vmin=None, vmax=None):
    nodes = field.nodes[irho, itheta, iphi]
    xx = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
    yy = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])

    qmesh = ax.pcolormesh(
        xx, 
        yy, 
        field.values[irho, itheta, iphi], 
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="gouraud"
    )

    theta = np.linspace(field.min_coords[2], field.max_coords[2], 64)
    for boundary in boundaries:
        ax.plot(
            boundary * np.cos(theta),
            boundary * np.sin(theta),
            color="k",
            linewidth=1
        )

    return (qmesh)
    
    
plt.close("all")
fig, axes = plt.subplots(figsize=(12, 6), nrows=1, ncols=3)
for ax in axes:
    ax.set_aspect(1)
    
axes[0].set_title("Velocity")
qmesh = plot_field(solvers[0].vv, axes[0], itheta=0)
cbar = fig.colorbar(qmesh, ax=axes[0], orientation="horizontal")
cbar.set_label("$v_P$ (km s$^{-1}$)")
xx = src_loc[0] * np.cos(src_loc[2])
yy = src_loc[0] * np.sin(src_loc[2])
axes[0].scatter(xx, yy, marker="*", facecolor="w", edgecolor="k", linewidth=1, s=256, zorder=100)

axes[1].set_title("Downgoing wavefront")
qmesh = plot_field(solvers[0].tt, axes[1], itheta=0, cmap=plt.get_cmap("nipy_spectral_r"), vmin=0, vmax=solvers[-1].tt.values.max())
cbar = fig.colorbar(qmesh, ax=axes[1], orientation="horizontal")
cbar.set_label("Traveltime (s)")
axes[1].set_yticklabels([])

axes[2].set_title("Upgoing (reflected) wavefront")
qmesh = plot_field(solvers[1].tt, axes[2], itheta=0, cmap=plt.get_cmap("nipy_spectral_r"), vmin=0, vmax=solvers[-1].tt.values.max())
cbar = fig.colorbar(qmesh, ax=axes[2], orientation="horizontal")
cbar.set_label("Traveltime (s)")
axes[2].yaxis.tick_right()

for ax in axes:
    ax.tick_params(axis="y", left=True, right=True)
    
rx_loc = (6371, np.pi/2, src_loc[2] - np.pi/4)

for rx_phi in src_loc[2] + np.array([7, 5, 3, 1, -2, -4, -6]) * np.pi/16:
    # Define a receiver location.
    rx_loc = (6371, np.pi/2, rx_phi)

    # Create an empty raypath.
    ray = np.empty((0, 3))

    # Trace the ray from the receiver to the inner-core-boundary.
    ipath = len(solvers) - 1 # Start with the last segment of the propagation path.
    path_seg = path[ipath]
    solver = solvers[ipath]
    ray_seg = solver.trace_ray(np.array(rx_loc))
    
    # The ray refracts along the lower (i.e., inner-core) boundary, so this is
    # a hack to drop the refracted part of the raypath.
    idx = np.where((ray_seg[...,0] - path_seg[0]) < solver.tt.step_size * 3 )[0][-1]
    ray = np.vstack([ray, ray_seg[idx:]])

    # Trace the ray from the inner-core boundary to the source.
    ipath -= 1
    solver = solvers[ipath]
    ray_seg = solver.trace_ray(ray[0])
    ray = np.vstack([ray_seg, ray])
    
    
    # Plot the raypath on the velocity model.
    xx = ray[..., 0] * np.cos(ray[..., 2])
    yy = ray[..., 0] * np.sin(ray[..., 2])
    axes[0].plot(xx, yy, color="k", linewidth=0.5)
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:36:49 2024

@author: tnye
"""

# Imports
import numpy as np
import pandas as pd
import pykonal
import pyproj
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define grid parameters
nx, ny, nz = 11, 11, 11
dx, dy, dz = 1, 1, 1  # Node intervals in meters

Vp = [2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]

X, Y, Z = np.meshgrid(np.arange(0,10.5,dx), np.arange(0,10.5,dy), np.arange(0,10.5,dz), indexing='ij')
Z = -1*Z

V = np.zeros((nx, ny, nz), dtype=int)
for x in range(nx):
    for y in range(ny):
        for z in range(nz):
            V[x,y,z] = Vp[z]
            
            
#%%


# Create a PyKonal solver instance for Cartesian coordinates
solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")

# Define the computational domain in Cartesian coordinates
solver.velocity.min_coords = [np.min(X.ravel()), np.min(Y.ravel()), np.min(Z.ravel())]  # Adjust minimum coordinates as per your grid
solver.velocity.node_intervals = [dx, dy, dz]
solver.velocity.npts = [nx, ny, nz]  # Number of nodes in x, y, z

# Example velocity model (uniform velocity for demonstration)
solver.velocity.values = V  # Assign velocity values to the solver

# Define the source location in Cartesian coordinates
src_loc = np.array([2.5,  0.0, -4.0], dtype=np.float64)

# Define the receiver location at the surface

# Set the source location for the solver
solver.src_loc = src_loc

# Solve the Eikonal equation for the upgoing ray to the receiver location
print("Solving the Eikonal equation for the upgoing ray...")
solver.solve()

# Define the receiver location at the surface
rx_loc = np.array([5.0, 0.0, 0.0], dtype=np.float64)

# Trace the ray from the receiver to the source
print("Tracing the ray from the receiver to the source...")
ray = solver.trace_ray(rx_loc)


#%%
# Plot the raypath
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=V, cmap='plasma_r', marker='.',alpha=0.3)
ax.scatter(src_loc[0], src_loc[1], src_loc[2], marker='*', c='goldenrod', s=200)
ax.scatter(rx_loc[0], rx_loc[1], rx_loc[2], marker='^', c='green', s=150)
mappable = plt.cm.ScalarMappable(cmap='plasma_r')
mappable.set_array(V)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('V')
ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color='k', linewidth=2)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Ray Path')


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X.ravel(), Z.ravel(), c=V, cmap='plasma_r', marker='.',alpha=0.3)
ax.scatter(src_loc[0], src_loc[2], marker='*', c='goldenrod', s=200)
ax.scatter(rx_loc[0], rx_loc[2], marker='^', c='green', s=150)
mappable = plt.cm.ScalarMappable(cmap='plasma_r')
mappable.set_array(V)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('V')
ax.plot(ray[:, 0], ray[:, 2], color='k', linewidth=2)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Ray Path')

#%%

# Imports
import numpy as np
import pandas as pd
import pykonal
import pyproj
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load IASP91 model for reference (if needed, though it's not used in this example)
iasp91 = pd.read_csv(
    '/Users/tnye/Downloads/IASP91.csv',
    header=None,
    names=["depth", "radius", "Vp", "Vs"]
)

# Define grid parameters with even higher resolution
nx, ny, nz = 101, 101, 101  # Increase the number of grid points significantly
dx, dy, dz = 0.1, 0.1, 0.1  # Decrease the node intervals further for even higher resolution

Vp = [2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]

X, Y, Z = np.meshgrid(np.arange(0, 10.1, dx), np.arange(0, 10.1, dy), np.arange(0, 10.1, dz), indexing='ij')
Z = -1 * Z

V = np.zeros((nx, ny, nz), dtype=float)
for x in range(nx):
    for y in range(ny):
        for z in range(nz):
            V[x, y, z] = Vp[z // 5]  # Adjusting for new grid resolution

# Create a PyKonal solver instance for Cartesian coordinates
solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")

# Define the computational domain in Cartesian coordinates
solver.velocity.min_coords = [np.min(X.ravel()), np.min(Y.ravel()), np.min(Z.ravel())]  # Adjust minimum coordinates as per your grid
solver.velocity.node_intervals = [dx, dy, dz]
solver.velocity.npts = [nx, ny, nz]  # Number of nodes in x, y, z

# Assign velocity values to the solver
solver.velocity.values = V

# Define the source location in Cartesian coordinates
src_loc = np.array([2.0, 0.0, -4.0], dtype=np.float64)

# Set the source location for the solver
solver.src_loc = src_loc

# Solve the Eikonal equation for the upgoing ray to the receiver location
print("Solving the Eikonal equation for the upgoing ray...")
solver.solve()

# Define the receiver location at the surface
rx_loc = np.array([5.0, 0.0, 0.0], dtype=np.float64)

# Trace the ray from the receiver to the source
print("Tracing the ray from the receiver to the source...")
ray = solver.trace_ray(rx_loc)

# Print the traced ray
print("Traced Ray Points:\n", ray)

# Plot the velocity model and the ray path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=V, cmap='plasma_r', marker='.', alpha=0.3)
ax.scatter(src_loc[0], src_loc[1], src_loc[2], marker='*', c='goldenrod', s=200)
ax.scatter(rx_loc[0], rx_loc[1], rx_loc[2], marker='^', c='green', s=150)
mappable = plt.cm.ScalarMappable(cmap='plasma_r')
mappable.set_array(V)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('V')
ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color='k', linewidth=2)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Ray Path')
plt.show()



#%%

# Imports
import numpy as np
import pandas as pd
import pykonal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator

# Define grid parameters with higher resolution
nx, ny, nz = 101, 101, 101  # Increase the number of grid points significantly
dx, dy, dz = 0.1, 0.1, 0.1  # Node intervals in meters

# Create the grid
X, Y, Z = np.meshgrid(np.arange(0, 10.1, dx), np.arange(0, 10.1, dy), np.arange(0, 10.1, dz), indexing='ij')

# Create a complex velocity model with increasing velocity at greater depths
def create_velocity_model(X, Y, Z):
    nx, ny, nz = X.shape
    V = np.zeros((nx, ny, nz), dtype=float)
    
    # Define a base velocity gradient (velocity increases with depth)
    base_velocity = 2.0 + 4.0 * (Z - Z.min()) / (Z.max() - Z.min())

    # Add random variations to simulate heterogeneity
    np.random.seed(42)  # For reproducibility
    random_variations = 0.5 * np.random.randn(nx, ny, nz)

    # Combine base gradient and random variations
    V = base_velocity + random_variations
    return V

V = create_velocity_model(X, Y, Z)

# Plot the velocity model slice at y = 0
fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(X[:, 0, :].ravel(), Z[:, 0, :].ravel(), c=V[:, 0, :].ravel(), cmap='plasma_r', marker='.')
mappable = plt.cm.ScalarMappable(cmap='plasma_r')
mappable.set_array(V[:, 0, :].ravel())
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Velocity (km/s)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Depth (m)')
ax.set_title('Velocity Model Slice at Y=0')
ax.invert_yaxis()
plt.show()

# Create a PyKonal solver instance for Cartesian coordinates
solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")

# Define the computational domain in Cartesian coordinates
solver.velocity.min_coords = [np.min(X.ravel()), np.min(Y.ravel()), np.min(Z.ravel())]  # Adjust minimum coordinates as per your grid
solver.velocity.node_intervals = [dx, dy, dz]
solver.velocity.npts = [nx, ny, nz]  # Number of nodes in x, y, z

# Assign velocity values to the solver
solver.velocity.values = V

# Define the source location in Cartesian coordinates
src_loc = np.array([2.7, 3.2, 6.1], dtype=np.float64)  # Note: depth is positive now

# Set the source location for the solver
solver.src_loc = src_loc

# Solve the Eikonal equation for the upgoing ray to the receiver location
print("Solving the Eikonal equation for the upgoing ray...")
solver.solve()

# Define the receiver location at the surface
rx_loc = np.array([5.1, 7.7, 0.0], dtype=np.float64)

# Trace the ray from the receiver to the source
print("Tracing the ray from the receiver to the source...")
ray = solver.trace_ray(rx_loc)

# Plot the ray path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=V.ravel(), cmap='plasma_r', marker='.', alpha=0.1)¡
ax.scatter(src_loc[0], src_loc[1], src_loc[2], marker='*', c='goldenrod', s=200)
ax.scatter(rx_loc[0], rx_loc[1], rx_loc[2], marker='^', c='green', s=150)
mappable = plt.cm.ScalarMappable(cmap='plasma_r')
mappable.set_array(V.ravel())
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Velocity (km/s)')
ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color='k', linewidth=2)
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_zlim(0,10)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Depth (m)')
ax.set_title('Ray Path')
ax.invert_yaxis()
plt.show()

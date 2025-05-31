import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from collections import deque
from scipy.interpolate import splprep, splev

# === Load Data ===
file_path = "main/robot_output/trajectory_lidar_data.npz"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit(1)

data = np.load(file_path)
x_traj = data['x_traj']
y_traj = data['y_traj']
lidar_x = data['lidar_x']
lidar_y = data['lidar_y']
points = np.vstack((lidar_x, lidar_y)).T

# === Step 1: Grid Setup ===
resolution = 0.05  # 5cm cells
margin = 1.0  # map size: 2m x 2m
grid_size = int(2 * margin / resolution)
grid = np.zeros((grid_size, grid_size), dtype=bool)
origin = -margin

def world_to_grid(x, y):
    gx = int((x - origin) / resolution)
    gy = int((y - origin) / resolution)
    return gx, gy

def grid_to_world(gx, gy):
    x = gx * resolution + origin + resolution / 2
    y = gy * resolution + origin + resolution / 2
    return x, y

# === Step 2: Mark occupied cells as True ===
for x, y in points:
    gx, gy = world_to_grid(x, y)
    if 0 <= gx < grid_size and 0 <= gy < grid_size:
        grid[gx, gy] = True

# === Step 3: Find connected blobs ===
visited = np.zeros_like(grid, dtype=bool)
directions = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0)]
splines = []

for x in range(grid_size):
    for y in range(grid_size):
        if grid[x, y] and not visited[x, y]:
            # Start BFS
            queue = deque()
            queue.append((x, y))
            blob = []

            while queue:
                cx, cy = queue.popleft()
                if not (0 <= cx < grid_size and 0 <= cy < grid_size):
                    continue
                if visited[cx, cy] or not grid[cx, cy]:
                    continue

                visited[cx, cy] = True
                blob.append(grid_to_world(cx, cy))

                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if grid[nx, ny] and not visited[nx, ny]:
                            queue.append((nx, ny))

            if len(blob) >= 5:
                blob = np.array(blob)
                try:
                    tck, u = splprep(blob.T, s=0.01)
                    u_new = np.linspace(0, 1, 100)
                    x_spline, y_spline = splev(u_new, tck)
                    splines.append(np.vstack((x_spline, y_spline)).T)
                except:
                    splines.append(blob)  # fallback to polyline

# === Step 4: For every spline's two endpoints, find closest endpoint of another spline ===
frontiers = []  # collect multiple frontier connections

for i, spline_i in enumerate(splines):
    endpoints_i = [spline_i[0], spline_i[-1]]
    for endpoint_i in endpoints_i:
        min_dist = float('inf')
        closest_endpoint = None
        
        # Search all other splines' endpoints
        for j, spline_j in enumerate(splines):
            if i == j:
                continue  # skip same spline
            
            endpoints_j = [spline_j[0], spline_j[-1]]
            for endpoint_j in endpoints_j:
                dist = np.linalg.norm(endpoint_i - endpoint_j)
                if 0.1 < dist < min_dist:  # frontier shouldn't be too short
                    min_dist = dist
                    closest_endpoint = endpoint_j
        
        if closest_endpoint is not None:
            frontiers.append((endpoint_i, closest_endpoint))

# Remove duplicates if needed (optional)
unique_frontiers = []
seen = set()
for p1, p2 in frontiers:
    key = tuple(sorted((tuple(p1), tuple(p2))))
    if key not in seen:
        unique_frontiers.append((p1, p2))
        seen.add(key)

# === Plotting ===
plt.figure(figsize=(8, 6))
plt.plot(x_traj, y_traj, 'b-', label='Robot Trajectory', linewidth=1.2)
plt.scatter(points[:, 0], points[:, 1], s=5, c='gray', alpha=0.3, label='LiDAR Points')

for spline in splines:
    plt.plot(spline[:, 0], spline[:, 1], linewidth=1.5)

if frontiers:
    for p1, p2 in unique_frontiers:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', linewidth=2, label='Frontier')

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Grid-based Curve Fitting and Logical Frontier')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()

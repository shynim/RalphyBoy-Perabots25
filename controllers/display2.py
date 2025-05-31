import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from collections import deque
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree # For faster closest endpoint search

# --- Configuration ---
LIDAR_DATA_PATH = "main/robot_output/trajectory_lidar_data.npz" # Your data path
RESOLUTION = 0.005  # meters per cell (5cm)
MIN_BLOB_POINTS_FOR_SPLINE = 10 # Minimum number of grid cells in a blob to fit a spline
S_SPLINE_SMOOTHING_FACTOR = 10 # Smoothing factor for splprep (0=interpolate, >0 smooth)
MIN_FRONTIER_DISTANCE_METERS = 0.1 # Min distance for a frontier connection
MAX_FRONTIER_DISTANCE_METERS = 2.0 # Max distance for a frontier connection


# --- Helper Functions ---
def load_lidar_and_trajectory_data(filepath):
    """Loads Lidar points (Nx2) and trajectory (Mx2) from an .npz file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        print("Creating a DUMMY file: dummy_for_spline_frontiers.npz")
        rng = np.random.default_rng(0)
        
        t1 = np.linspace(0, np.pi, 30)
        x1 = np.cos(t1) * 1.0 + rng.normal(scale=0.05, size=t1.shape) - 1.5
        y1 = np.sin(t1) * 0.5 + rng.normal(scale=0.05, size=t1.shape)

        t2 = np.linspace(np.pi/2, 3*np.pi/2, 30)
        x2 = np.cos(t2) * 0.8 + rng.normal(scale=0.05, size=t2.shape) + 1.5
        y2 = np.sin(t2) * 0.6 + rng.normal(scale=0.05, size=t2.shape)
        
        lidar_x_dummy = np.concatenate([x1, x2])
        lidar_y_dummy = np.concatenate([y1, y2])
        
        traj_x_dummy = np.linspace(-2, 2, 50)
        traj_y_dummy = np.zeros_like(traj_x_dummy) * 0.1

        dummy_filepath = "dummy_for_spline_frontiers.npz"
        # Avoid overwriting if LIDAR_DATA_PATH is set to the dummy name and it already exists
        if os.path.abspath(filepath) == os.path.abspath(dummy_filepath) and os.path.exists(filepath):
             dummy_filepath = "dummy_for_spline_frontiers_alt.npz"

        np.savez(dummy_filepath, 
                 x_traj=traj_x_dummy, 
                 y_traj=traj_y_dummy,
                 lidar_x=lidar_x_dummy, 
                 lidar_y=lidar_y_dummy)
        filepath = dummy_filepath 
        print(f"Using DUMMY data from: {filepath}")
        
    try:
        data = np.load(filepath)
        x_traj_data = data['x_traj']
        y_traj_data = data['y_traj']
        lidar_x_data = data['lidar_x']
        lidar_y_data = data['lidar_y']
        
        robot_trajectory = np.column_stack((x_traj_data, y_traj_data))
        lidar_points = np.column_stack((lidar_x_data, lidar_y_data))
        
        print(f"Loaded {len(lidar_points)} lidar points and {len(robot_trajectory)} trajectory points from {filepath}.")
        return lidar_points, robot_trajectory
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None, None

class OccupancyGridManager:
    def __init__(self, points_for_bounds, resolution, margin_percentage=0.1):
        self.resolution = resolution
        
        if points_for_bounds is None or len(points_for_bounds) == 0:
            print("WARN: OccupancyGridManager initialized with no points for bounds. Using default small grid.")
            self.origin_x = -1.0
            self.origin_y = -1.0
            self.world_max_x = 1.0
            self.world_max_y = 1.0
        else:
            min_x, min_y = np.min(points_for_bounds, axis=0)
            max_x, max_y = np.max(points_for_bounds, axis=0)

            range_x = max_x - min_x
            range_y = max_y - min_y
            margin_x = max(self.resolution * 5, range_x * margin_percentage) 
            margin_y = max(self.resolution * 5, range_y * margin_percentage)

            self.origin_x = min_x - margin_x
            self.origin_y = min_y - margin_y
            self.world_max_x = max_x + margin_x
            self.world_max_y = max_y + margin_y

        self.grid_width_cells = int(np.ceil((self.world_max_x - self.origin_x) / self.resolution))
        self.grid_height_cells = int(np.ceil((self.world_max_y - self.origin_y) / self.resolution))

        if self.grid_width_cells <= 0: self.grid_width_cells = 10 
        if self.grid_height_cells <= 0: self.grid_height_cells = 10

        # Grid: True for occupied, False for free. Stored as (cols, rows) or (width_idx, height_idx)
        self.grid = np.zeros((self.grid_width_cells, self.grid_height_cells), dtype=bool)
        
    def populate_grid_with_points(self, points_to_occupy):
        if points_to_occupy is None or len(points_to_occupy) == 0:
            print("WARN: No points provided to populate grid occupancy.")
            return
            
        for x_w, y_w in points_to_occupy:
            gx, gy = self.world_to_grid_coords(x_w, y_w)
            if 0 <= gx < self.grid_width_cells and 0 <= gy < self.grid_height_cells:
                self.grid[gx, gy] = True # grid[col_idx, row_idx]
    
    def world_to_grid_coords(self, world_x, world_y):
        gx = int((world_x - self.origin_x) / self.resolution)
        gy = int((world_y - self.origin_y) / self.resolution)
        return gx, gy # (col_idx, row_idx)

    def grid_to_world_coords(self, grid_x, grid_y): # grid_x is col_idx, grid_y is row_idx
        world_x = self.origin_x + (grid_x + 0.5) * self.resolution
        world_y = self.origin_y + (grid_y + 0.5) * self.resolution
        return world_x, world_y

    def get_grid_for_display(self):
        # Returns a version for imshow (0 for occupied, 255 for free)
        # imshow expects (rows, cols)
        display_grid_for_imshow = np.full((self.grid_height_cells, self.grid_width_cells), 255, dtype=np.uint8)
        
        # self.grid is (width_cells, height_cells) i.e. (cols, rows)
        # self.grid.T is (height_cells, width_cells) i.e. (rows, cols)
        # This matches display_grid_for_imshow's shape.
        display_grid_for_imshow[self.grid.T] = 0 
        return display_grid_for_imshow


def find_occupied_blobs_and_fit_splines(grid_manager):
    grid = grid_manager.grid
    grid_width = grid_manager.grid_width_cells
    grid_height = grid_manager.grid_height_cells
    
    visited = np.zeros_like(grid, dtype=bool)
    directions = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0)]
    
    object_splines = [] 

    for r_idx in range(grid_height): 
        for c_idx in range(grid_width): 
            if grid[c_idx, r_idx] and not visited[c_idx, r_idx]:
                queue = deque()
                queue.append((c_idx, r_idx)) 
                blob_grid_coords = [] 

                while queue:
                    curr_gx, curr_gy = queue.popleft()
                    if not (0 <= curr_gx < grid_width and 0 <= curr_gy < grid_height):
                        continue
                    if visited[curr_gx, curr_gy] or not grid[curr_gx, curr_gy]:
                        continue
                    visited[curr_gx, curr_gy] = True
                    blob_grid_coords.append((curr_gx, curr_gy))
                    for dr, dc in directions: 
                        next_gx, next_gy = curr_gx + dc, curr_gy + dr
                        if 0 <= next_gx < grid_width and 0 <= next_gy < grid_height:
                            if grid[next_gx, next_gy] and not visited[next_gx, next_gy]:
                                queue.append((next_gx, next_gy))
                
                if len(blob_grid_coords) >= MIN_BLOB_POINTS_FOR_SPLINE:
                    blob_world_coords = np.array([grid_manager.grid_to_world_coords(gx, gy) for gx, gy in blob_grid_coords])
                    
                    # Attempt to order points roughly for better spline fit.
                    # This is a simple heuristic: sort by X, then find a path.
                    # More sophisticated methods (like finding the perimeter) could be used.
                    if len(blob_world_coords) > 1:
                        # Simple sort by x, then y (might not be ideal for all shapes)
                        # blob_world_coords = blob_world_coords[np.lexsort((blob_world_coords[:,1], blob_world_coords[:,0]))]
                        # A slightly better approach for blobby shapes could be to start from an
                        # "extreme" point and iteratively add the closest unvisited point.
                        # For now, relying on BFS order and spline smoothing.
                        pass


                    try:
                        k_spline = min(3, len(blob_world_coords) - 1)
                        if k_spline < 1: k_spline = 1 

                        tck, u = splprep(blob_world_coords.T, s=S_SPLINE_SMOOTHING_FACTOR, k=k_spline, nest=-1, quiet=2) # quiet=2 suppresses messages
                        u_new = np.linspace(u.min(), u.max(), 100) 
                        x_spline, y_spline = splev(u_new, tck)
                        object_splines.append(np.vstack((x_spline, y_spline)).T)
                    except Exception as e:
                        # print(f"Spline fitting failed: {e}. Using raw blob points for blob of size {len(blob_world_coords)}.")
                        object_splines.append(blob_world_coords) # Fallback to raw points
                        
    return object_splines


def find_frontiers_between_splines(object_splines):
    if not object_splines or len(object_splines) < 2:
        return []

    all_endpoints = []
    spline_indices = [] 
    for i, spline in enumerate(object_splines):
        if len(spline) > 0:
            all_endpoints.append(spline[0]) 
            spline_indices.append(i)
            if len(spline) > 1: 
                 all_endpoints.append(spline[-1]) 
                 spline_indices.append(i)
    
    if not all_endpoints or len(all_endpoints) < 2 :
        return []

    all_endpoints_np = np.array(all_endpoints)
    kdtree = KDTree(all_endpoints_np)
    
    frontiers = []
    processed_pairs = set() 

    for i, p1 in enumerate(all_endpoints_np):
        p1_spline_idx = spline_indices[i]
        
        k_query = min(len(all_endpoints_np), len(object_splines) * 2) # Max k needed
        if k_query <= 1 and len(all_endpoints_np) > 1: # Need at least 2 for self + other
            k_query = 2
        elif k_query == 0:
            continue

        distances, indices = kdtree.query(p1, k=k_query)

        if np.isscalar(distances): distances = [distances]
        if np.isscalar(indices): indices = [indices]

        for j_loop_idx in range(len(indices)): # Renamed loop variable
            p2_idx_in_all_endpoints = indices[j_loop_idx]
            dist = distances[j_loop_idx]

            if p2_idx_in_all_endpoints == i: 
                continue

            p2 = all_endpoints_np[p2_idx_in_all_endpoints]
            p2_spline_idx = spline_indices[p2_idx_in_all_endpoints]

            if p1_spline_idx == p2_spline_idx: 
                continue

            if MIN_FRONTIER_DISTANCE_METERS < dist < MAX_FRONTIER_DISTANCE_METERS:
                p1_tuple = tuple(np.round(p1, 5)) # Round for consistent hashing
                p2_tuple = tuple(np.round(p2, 5))
                pair_key = tuple(sorted((p1_tuple, p2_tuple)))
                if pair_key not in processed_pairs:
                    frontiers.append((p1, p2)) # Store original precision points
                    processed_pairs.add(pair_key)
                break 
    
    return frontiers

# --- Main Execution ---
def main():
    lidar_points, robot_trajectory = load_lidar_and_trajectory_data(LIDAR_DATA_PATH)
    if lidar_points is None:
        sys.exit(1)

    # 1. Create Occupancy Grid
    print("Creating occupancy grid...")
    points_for_bounds = lidar_points
    if robot_trajectory is not None and len(robot_trajectory) > 0:
        if points_for_bounds is not None and len(points_for_bounds) > 0:
            points_for_bounds = np.vstack((points_for_bounds, robot_trajectory))
        else:
            points_for_bounds = robot_trajectory # Only trajectory if no lidar for bounds
            
    if points_for_bounds is None or not points_for_bounds.any(): # .any() checks if array is not empty
        grid_manager = OccupancyGridManager(None, RESOLUTION) # Uses default bounds
    else:
        grid_manager = OccupancyGridManager(points_for_bounds, RESOLUTION)

    # Populate the grid with Lidar points for occupancy
    if lidar_points is not None and len(lidar_points) > 0:
        grid_manager.populate_grid_with_points(lidar_points)
    else:
        print("WARN: No Lidar points provided to populate the occupancy grid.")

    # 2. Find occupied blobs and fit splines
    print("Finding occupied blobs and fitting splines...")
    object_splines = find_occupied_blobs_and_fit_splines(grid_manager)
    print(f"Found {len(object_splines)} distinct object splines.")

    # 3. Find frontiers by connecting spline endpoints
    print("Finding frontiers between object splines...")
    frontiers = find_frontiers_between_splines(object_splines)
    print(f"Found {len(frontiers)} frontier connections.")

    if frontiers:
        print("\n--- Detected Frontier Connections ---")
        for idx, (p1, p2) in enumerate(frontiers):
            print(f"frontier {idx+1} ({p1[0]:.3f},{p1[1]:.3f}) ({p2[0]:.3f},{p2[1]:.3f})")
        print("-----------------------------------\n")

    # 4. Plotting
    print("Visualizing...")
    plt.figure(figsize=(12, 10))
    
    display_grid_img = grid_manager.get_grid_for_display()
    plt.imshow(display_grid_img, cmap='gray_r', origin='lower', 
               extent=[grid_manager.origin_x, grid_manager.origin_x + grid_manager.grid_width_cells * RESOLUTION,
                       grid_manager.origin_y, grid_manager.origin_y + grid_manager.grid_height_cells * RESOLUTION],
               interpolation='nearest', alpha=0.3)

    if lidar_points is not None and len(lidar_points) > 0:
        plt.scatter(lidar_points[:, 0], lidar_points[:, 1], s=1, c='dimgray', alpha=0.5, label='Lidar Points')

    if robot_trajectory is not None and len(robot_trajectory) > 0:
        plt.plot(robot_trajectory[:, 0], robot_trajectory[:, 1], 'b-', linewidth=1.5, label='Robot Trajectory')
        plt.plot(robot_trajectory[0, 0], robot_trajectory[0, 1], 'go', markersize=5, label='Start')
        plt.plot(robot_trajectory[-1, 0], robot_trajectory[-1, 1], 'ro', markersize=5, label='End')

    for i, spline_points in enumerate(object_splines):
        if len(spline_points) > 1:
            plt.plot(spline_points[:, 0], spline_points[:, 1], 'k-', linewidth=2, 
                     label='Object Splines' if i == 0 else None)
        elif len(spline_points) == 1: 
            plt.plot(spline_points[0,0], spline_points[0,1], 'ko', markersize=3)

    for i, (p1, p2) in enumerate(frontiers):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=1.5, 
                 label='Frontiers' if i == 0 else None)

    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Object Splines and Endpoint-Based Frontiers')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    plt.show()

if __name__ == "__main__":
    output_dir_for_dummy = os.path.dirname(LIDAR_DATA_PATH)
    if not os.path.exists(output_dir_for_dummy) and output_dir_for_dummy: # Ensure dir exists if path is relative
        os.makedirs(output_dir_for_dummy, exist_ok=True)
    
    if not os.path.exists(LIDAR_DATA_PATH):
         print(f"Note: {LIDAR_DATA_PATH} not found. `load_lidar_and_trajectory_data` will generate a dummy file.")

    main()
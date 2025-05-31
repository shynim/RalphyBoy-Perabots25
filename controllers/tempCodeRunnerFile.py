import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import KDTree
import os

# --- Configuration ---
LIDAR_DATA_PATH = "main/robot_output/trajectory_lidar_data.npz"
CELL_SIZE = 0.005  # 1 cm in meters
MIN_WALL_LENGTH_METERS = 0.5 # Minimum length for a segment to be considered a wall (5cm)
CONTOUR_APPROX_EPSILON_FACTOR = 0.005 # Adjusted for potentially straighter walls

# --- Helper Functions ---

def load_data(filepath):
    """Loads lidar data and trajectory from .npz file based on specified keys."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        print("Please ensure the path is correct or create a dummy file for testing.")
        print("Creating a dummy file now: dummy_trajectory_lidar_data.npz with new keys.")
        # Create dummy data for demonstration if file not found
        rng = np.random.default_rng(0)
        points = []
        # Room walls (e.g., 5m x 5m)
        for x_val in np.linspace(-2.5, 2.5, 100):
            points.append([x_val, -2.5]) # Bottom wall
            points.append([x_val, 2.5])  # Top wall
        for y_val in np.linspace(-2.5, 2.5, 100):
            points.append([-2.5, y_val]) # Left wall
            points.append([2.5, y_val])  # Right wall
        # A box inside
        for x_val in np.linspace(-0.5, 0.5, 20):
            points.append([x_val, -0.5])
            points.append([x_val, 0.5])
        for y_val in np.linspace(-0.5, 0.5, 20):
            points.append([-0.5, y_val])
            points.append([0.5, y_val])
        
        lidar_points_world_dummy = np.array(points) + rng.normal(scale=0.02, size=(len(points), 2))
        
        # Simulate a simple trajectory
        trajectory_x_dummy = np.linspace(-2, 2, 50)
        trajectory_y_dummy = np.zeros_like(trajectory_x_dummy)
        
        dummy_filepath = "dummy_trajectory_lidar_data.npz" # Save in current dir for simplicity
        np.savez(dummy_filepath, 
                 x_traj=trajectory_x_dummy,
                 y_traj=trajectory_y_dummy,
                 lidar_x=lidar_points_world_dummy[:, 0],
                 lidar_y=lidar_points_world_dummy[:, 1])
        filepath = dummy_filepath
        print(f"Using dummy data from {filepath}")

    try:
        data = np.load(filepath, allow_pickle=True)
        
        x_traj_data = data.get('x_traj')
        y_traj_data = data.get('y_traj')
        lidar_x_data = data.get('lidar_x')
        lidar_y_data = data.get('lidar_y')

        if x_traj_data is None or y_traj_data is None:
            raise ValueError("Keys 'x_traj' or 'y_traj' not found in NPZ file.")
        if lidar_x_data is None or lidar_y_data is None:
            raise ValueError("Keys 'lidar_x' or 'lidar_y' not found in NPZ file.")

        # Reconstruct robot_trajectory (Nx2)
        robot_trajectory = np.column_stack((x_traj_data, y_traj_data))
        
        # Reconstruct lidar_points (Nx2)
        # This assumes lidar_x_data and lidar_y_data are 1D arrays of the same length
        if lidar_x_data.ndim == 0: # Handle scalar case if only one point
             lidar_x_data = np.array([lidar_x_data])
        if lidar_y_data.ndim == 0:
             lidar_y_data = np.array([lidar_y_data])
             
        if len(lidar_x_data) != len(lidar_y_data):
            raise ValueError(f"Lidar x ({len(lidar_x_data)}) and y ({len(lidar_y_data)}) data have mismatched lengths.")
        
        lidar_points = np.column_stack((lidar_x_data, lidar_y_data))
            
        if lidar_points.ndim != 2 or lidar_points.shape[1] != 2:
            raise ValueError(f"Reconstructed Lidar points have unexpected shape: {lidar_points.shape}. Expected (N, 2).")
        if robot_trajectory.ndim != 2 or robot_trajectory.shape[1] != 2: # Assuming (x,y) only now
             raise ValueError(f"Reconstructed Robot trajectory has unexpected shape: {robot_trajectory.shape}. Expected (N, 2).")

        print(f"Loaded {len(lidar_points)} lidar points and {len(robot_trajectory)} trajectory points.")
        return lidar_points, robot_trajectory
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def create_occupancy_grid(points, cell_size):
    """Creates a B&W occupancy grid from points."""
    if points is None or len(points) == 0:
        return None, 0, 0, 0, 0

    x_min, y_min = np.min(points, axis=0) - 2 * cell_size # Add some padding
    x_max, y_max = np.max(points, axis=0) + 2 * cell_size

    grid_width = int(np.ceil((x_max - x_min) / cell_size))
    grid_height = int(np.ceil((y_max - y_min) / cell_size))

    grid = np.full((grid_height, grid_width), 255, dtype=np.uint8)

    for point in points:
        gx = int((point[0] - x_min) / cell_size)
        gy = int((point[1] - y_min) / cell_size) 

        if 0 <= gx < grid_width and 0 <= gy < grid_height:
            grid[gy, gx] = 0 

    return grid, x_min, y_min, grid_width, grid_height

def grid_to_world(gx, gy, x_min_grid, y_min_grid, cell_size, grid_height_pixels=None):
    """Converts grid pixel coordinates to world coordinates (center of pixel)."""
    wy = y_min_grid + (gy + 0.5) * cell_size
    wx = x_min_grid + (gx + 0.5) * cell_size
    return wx, wy

def detect_wall_segments(grid_image, x_min_grid, y_min_grid, cell_size, min_segment_len_pixels):
    """Detects wall segments from the occupancy grid image."""
    contours, _ = cv2.findContours(grid_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wall_segments_world = []
    all_endpoints_world = set()

    for_cv_vis = None
    # if __debug__: # Only create if debugging or showing
    #     for_cv_vis = grid_image.copy()
    #     if len(for_cv_vis.shape) == 2: 
    #         for_cv_vis = cv2.cvtColor(for_cv_vis, cv2.COLOR_GRAY2BGR)

    for i, cnt in enumerate(contours):
        epsilon = CONTOUR_APPROX_EPSILON_FACTOR * cv2.arcLength(cnt, True)
        approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
        
        # if for_cv_vis is not None:
        #     cv2.drawContours(for_cv_vis, [approx_poly], -1, (0,0,255), 1) 

        if len(approx_poly) >= 2: 
            for j in range(len(approx_poly) -1): # Iterate through segments of the polygon
                p1_grid = (approx_poly[j][0][0], approx_poly[j][0][1]) 
                p2_grid = (approx_poly[j+1][0][0], approx_poly[j+1][0][1])

                if np.linalg.norm(np.array(p1_grid) - np.array(p2_grid)) < min_segment_len_pixels:
                    continue

                p1_world = grid_to_world(p1_grid[0], p1_grid[1], x_min_grid, y_min_grid, cell_size)
                p2_world = grid_to_world(p2_grid[0], p2_grid[1], x_min_grid, y_min_grid, cell_size)
                
                wall_segments_world.append( (p1_world, p2_world) )
                all_endpoints_world.add(p1_world)
                all_endpoints_world.add(p2_world)

            # For closed polygons, also add segment from last point to first point
            if len(approx_poly) > 2: 
                p_last_grid = (approx_poly[-1][0][0], approx_poly[-1][0][1])
                p_first_grid = (approx_poly[0][0][0], approx_poly[0][0][1])
                # Check if it's reasonably closed; approxPolyDP might open it slightly
                # or if the original contour was not actually closed.
                # For RETR_EXTERNAL, usually closed.
                if np.linalg.norm(np.array(p_last_grid) - np.array(p_first_grid)) >= min_segment_len_pixels:
                    p_last_world = grid_to_world(p_last_grid[0], p_last_grid[1], x_min_grid, y_min_grid, cell_size)
                    p_first_world = grid_to_world(p_first_grid[0], p_first_grid[1], x_min_grid, y_min_grid, cell_size)
                    wall_segments_world.append( (p_last_world, p_first_world) )
                    all_endpoints_world.add(p_last_world)
                    all_endpoints_world.add(p_first_world)
    
    # if for_cv_vis is not None:
    #     cv2.imshow("Approximated Contours", for_cv_vis)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return wall_segments_world, list(all_endpoints_world)


def find_frontiers(unique_wall_endpoints_world, wall_segments_world):
    """Finds frontiers by connecting closest wall endpoints not already forming a wall."""
    if not unique_wall_endpoints_world or len(unique_wall_endpoints_world) < 2:
        return []

    endpoints_array = np.array(list(unique_wall_endpoints_world))
    kdtree = KDTree(endpoints_array)

    wall_set = set()
    for p1, p2 in wall_segments_world:
        p1_tuple = tuple(p1)
        p2_tuple = tuple(p2)
        wall_set.add(frozenset({p1_tuple, p2_tuple}))

    frontiers = []
    processed_pairs = set() 

    for i, p1_world_coords in enumerate(endpoints_array):
        p1_tuple = tuple(p1_world_coords)
        
        # Query for k=2 (itself and the closest other)
        # If fewer than 2 points total, kdtree.query might behave differently or error
        num_neighbors_to_query = min(2, len(endpoints_array))
        if num_neighbors_to_query < 2: # Need at least two points to form a pair
            continue

        distances, indices = kdtree.query(p1_world_coords, k=num_neighbors_to_query)

        if np.isscalar(indices) or len(indices) < 2: # Not enough distinct neighbors found
            continue
            
        p2_idx = indices[1] # Closest distinct neighbor
        p2_world_coords = endpoints_array[p2_idx]
        p2_tuple = tuple(p2_world_coords)

        if p1_tuple == p2_tuple: # Should not happen if indices[1] is used correctly
            continue

        current_pair_nodes = tuple(sorted((p1_tuple, p2_tuple), key=lambda p: (p[0], p[1])))
        current_pair_frozen = frozenset({p1_tuple, p2_tuple})

        if current_pair_frozen not in wall_set and current_pair_nodes not in processed_pairs:
            frontiers.append((p1_tuple, p2_tuple))
            processed_pairs.add(current_pair_nodes)
            
    return frontiers

# --- Main Visualization Logic ---
def main():
    lidar_points_raw, robot_trajectory = load_data(LIDAR_DATA_PATH)

    if lidar_points_raw is None or robot_trajectory is None:
        print("Failed to load data. Exiting.")
        return

    if len(robot_trajectory) > 0 and len(lidar_points_raw) > 0:
        all_points_for_bounds = np.vstack((lidar_points_raw, robot_trajectory))
    elif len(lidar_points_raw) > 0:
        all_points_for_bounds = lidar_points_raw
    elif len(robot_trajectory) > 0:
        all_points_for_bounds = robot_trajectory
    else:
        print("No points available (neither Lidar nor trajectory) to create a map. Exiting.")
        return

    # 1. Create Occupancy Grid
    print("Creating occupancy grid...")
    occupancy_grid, x_grid_origin, y_grid_origin, grid_w_pixels, grid_h_pixels = \
        create_occupancy_grid(lidar_points_raw, CELL_SIZE) # Use only lidar for grid
    
    if occupancy_grid is None:
        print("Failed to create occupancy grid.")
        return
    
    print(f"Grid created: {grid_w_pixels}x{grid_h_pixels} pixels. Origin: ({x_grid_origin:.2f}, {y_grid_origin:.2f})m")

    # 2. Detect Wall Segments
    print("Detecting wall segments...")
    min_wall_len_px = int(MIN_WALL_LENGTH_METERS / CELL_SIZE)
    wall_segments, unique_endpoints = detect_wall_segments(
        occupancy_grid, x_grid_origin, y_grid_origin, CELL_SIZE, min_wall_len_px
    )
    print(f"Detected {len(wall_segments)} wall segments and {len(unique_endpoints)} unique endpoints.")

    # 3. Find Frontiers
    print("Finding frontiers...")
    frontiers = find_frontiers(unique_endpoints, wall_segments)
    print(f"Found {len(frontiers)} frontier lines.")

    # 4. Visualize
    print("Visualizing...")
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.imshow(occupancy_grid, cmap='gray_r', origin='lower', 
              extent=[x_grid_origin, x_grid_origin + grid_w_pixels * CELL_SIZE,
                      y_grid_origin, y_grid_origin + grid_h_pixels * CELL_SIZE],
              interpolation='nearest', alpha=0.3)

    # if len(lidar_points_raw) > 0:
    #     ax.scatter(lidar_points_raw[:, 0], lidar_points_raw[:, 1], s=0.1, c='gray', alpha=0.5, label='Lidar Points')

    if len(robot_trajectory) > 0:
        ax.plot(robot_trajectory[:, 0], robot_trajectory[:, 1], 'b-', linewidth=1.5, label='Robot Trajectory')
        ax.plot(robot_trajectory[0, 0], robot_trajectory[0, 1], 'go', markersize=5, label='Start')
        ax.plot(robot_trajectory[-1, 0], robot_trajectory[-1, 1], 'ro', markersize=5, label='End')

    for i, (p1, p2) in enumerate(wall_segments):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=2, 
                label='Walls' if i == 0 else None)

    for i, (p1, p2) in enumerate(frontiers):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=1, 
                label='Frontiers' if i == 0 else None)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Lidar Data, Walls, Trajectory, and Frontiers Visualization")
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1)) # Adjusted for better legend placement
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()


if __name__ == "__main__":
    # Create dummy directory if it doesn't exist for the specified LIDAR_DATA_PATH
    output_dir_for_dummy = os.path.dirname(LIDAR_DATA_PATH)
    if not os.path.exists(output_dir_for_dummy) and output_dir_for_dummy: # Ensure output_dir is not empty
        os.makedirs(output_dir_for_dummy, exist_ok=True)
        print(f"Created directory: {output_dir_for_dummy}")
    
    # If LIDAR_DATA_PATH points to a non-existent file, dummy data will be created
    # in the current working directory as "dummy_trajectory_lidar_data.npz"
    # and then used.
    main()
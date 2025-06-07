import numpy as np
import cv2
import os
# import math # math is no longer needed

# --- Configuration ---
NPZ_FILE = 'robot_output/trajectory_lidar_data.npz'
MAP_SIZE_METERS = 2.0
CELL_SIZE_METERS = 0.005
GRID_DIM = int(MAP_SIZE_METERS / CELL_SIZE_METERS)

# Colors for drawing
COLOR_TRAJECTORY = (0, 255, 0)     # Green
COLOR_WALLS_ON_COSTMAP = (255, 255, 255) # White

# --- Helper: Create Dummy NPZ data ---
def create_dummy_data(filename):
    print(f"Creating dummy data: {filename}")
    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
            print(f"Created directory: {dir_name} for dummy data.")
        except OSError as e:
            print(f"Warning: Could not create directory {dir_name} for dummy data: {e}. Saving to current directory.")
            filename = os.path.basename(filename)

    x_traj = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3]) + 0.5
    y_traj = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]) + 0.5

    lidar_x_world, lidar_y_world = [], []
    # Box 1
    lidar_x_world.extend(np.linspace(0.1, 0.1, 20))
    lidar_y_world.extend(np.linspace(0.1, 0.8, 20))
    lidar_x_world.extend(np.linspace(0.3, 0.3, 20))
    lidar_y_world.extend(np.linspace(0.1, 0.8, 20))
    lidar_x_world.extend(np.linspace(0.1, 0.3, 10))
    lidar_y_world.extend(np.linspace(0.1, 0.1, 10))
    lidar_x_world.extend(np.linspace(0.1, 0.3, 10))
    lidar_y_world.extend(np.linspace(0.8, 0.8, 10))

    # Box 2 (rotated)
    for i in range(30):
        t = i / 29.0
        x = 0.6 + t * 0.1 # main length along x
        y = 0.3 + t * 0.4 # main length along y (slope)
        lidar_x_world.append(x)
        lidar_y_world.append(y)
        lidar_x_world.append(x + 0.05) # width
        lidar_y_world.append(y - 0.02) # width

    # Box 3
    lidar_x_world.extend(np.linspace(0.5, 0.8, 20))
    lidar_y_world.extend(np.linspace(0.9, 0.9, 20))
    lidar_x_world.extend(np.linspace(0.5, 0.8, 20))
    lidar_y_world.extend(np.linspace(1.1, 1.1, 20)) # 0.2m height
    lidar_x_world.extend(np.linspace(0.5,0.5,10))
    lidar_y_world.extend(np.linspace(0.9,1.1,10))
    lidar_x_world.extend(np.linspace(0.8,0.8,10))
    lidar_y_world.extend(np.linspace(0.9,1.1,10))

    lidar_x_world = np.array(lidar_x_world)
    lidar_y_world = np.array(lidar_y_world)
    np.savez(filename, x_traj=x_traj, y_traj=y_traj, lidar_x=lidar_x_world, lidar_y=lidar_y_world)
    print(f"Dummy data saved as {filename} with {len(lidar_x_world)} lidar points.")
    return x_traj, y_traj, lidar_x_world, lidar_y_world

# --- 1. Load Data ---
if not os.path.exists(NPZ_FILE):
    print(f"File {NPZ_FILE} not found. Creating dummy data for demonstration.")
    x_traj, y_traj, lidar_x_world, lidar_y_world = create_dummy_data(NPZ_FILE)
else:
    try:
        print(f"Loading data from {NPZ_FILE}...")
        data = np.load(NPZ_FILE)
        x_traj = data['x_traj']
        y_traj = data['y_traj']
        lidar_x_world = data['lidar_x']
        lidar_y_world = data['lidar_y']
        print(f"Data loaded: {len(lidar_x_world)} lidar points, {len(x_traj)} trajectory points.")
    except Exception as e:
        print(f"Error loading {NPZ_FILE}: {e}")
        print("Attempting to use dummy data instead, saved as 'dummy_fallback.npz'.")
        x_traj, y_traj, lidar_x_world, lidar_y_world = create_dummy_data("dummy_fallback.npz")

if lidar_x_world.size == 0:
    print("Lidar data is empty. Cannot proceed.")
    exit()

# --- 2. Determine Map Origin and Boundaries ---
data_min_x, data_max_x = np.min(lidar_x_world), np.max(lidar_x_world)
data_min_y, data_max_y = np.min(lidar_y_world), np.max(lidar_y_world)
data_center_x = (data_min_x + data_max_x) / 2.0
data_center_y = (data_min_y + data_max_y) / 2.0
MAP_ORIGIN_X_WORLD = data_center_x - (MAP_SIZE_METERS / 2.0)
MAP_ORIGIN_Y_WORLD = data_center_y - (MAP_SIZE_METERS / 2.0)
print(f"Map will cover world X: [{MAP_ORIGIN_X_WORLD:.2f}, {MAP_ORIGIN_X_WORLD + MAP_SIZE_METERS:.2f}]")
print(f"Map will cover world Y: [{MAP_ORIGIN_Y_WORLD:.2f}, {MAP_ORIGIN_Y_WORLD + MAP_SIZE_METERS:.2f}]")

# --- 3. Create 2D Grid Map ---
grid_map = np.zeros((GRID_DIM, GRID_DIM), dtype=np.uint8)

# --- 4. Populate Grid with Lidar Points ---
valid_lidar_points_count = 0
for wx, wy in zip(lidar_x_world, lidar_y_world):
    relative_x = wx - MAP_ORIGIN_X_WORLD
    relative_y = wy - MAP_ORIGIN_Y_WORLD
    grid_col = int(relative_x / CELL_SIZE_METERS)
    grid_row_raw = int(relative_y / CELL_SIZE_METERS)
    grid_row = GRID_DIM - 1 - grid_row_raw # Invert Y for typical image coordinates (origin top-left)
    if 0 <= grid_col < GRID_DIM and 0 <= grid_row < GRID_DIM:
        grid_map[grid_row, grid_col] = 255 # Mark wall pixels as white
        valid_lidar_points_count += 1
print(f"Populated grid with {valid_lidar_points_count} lidar points within map boundaries.")
if valid_lidar_points_count == 0:
    print("No lidar points fell within the defined map boundaries. Showing empty/minimal cost map.")

# --- 5. Generate Cost Map ---
# Create inverted map for distance transform (walls are 0, free space 255)
# This is because distanceTransform calculates distance to the nearest zero pixel.
inverted_grid_map = cv2.bitwise_not(grid_map)

# Calculate distance transform. Output is float32.
# For each pixel in inverted_grid_map, its distance to the nearest 0-pixel (wall) is computed.
dist_transform = cv2.distanceTransform(inverted_grid_map, cv2.DIST_L2, 5)

# Normalize the distance transform to range [0, 1] for visualization
# norm_dist_transform will be 0 at wall locations and 1 at the furthest point from any wall.
norm_dist_transform = np.zeros_like(dist_transform, dtype=np.float32)
if np.any(dist_transform): # Check if dist_transform is not all zeros to avoid division by zero with normalize
    cv2.normalize(dist_transform, norm_dist_transform, 0.0, 1.0, cv2.NORM_MINMAX)
# If dist_transform is all zeros (e.g., grid_map was all 255s), norm_dist_transform will remain all zeros.

# Create BGR image for cost map visualization
output_image = np.zeros((GRID_DIM, GRID_DIM, 3), dtype=np.uint8)

# Blue channel:
# We want high blue near walls (small distance, norm_dist_transform close to 0)
# and low blue far from walls (large distance, norm_dist_transform close to 1).
# So, blue_intensity = (1.0 - norm_dist_transform) * 255
blue_channel = ((1.0 - norm_dist_transform**0.5) * 255.0).astype(np.uint8)
output_image[:, :, 0] = blue_channel # Assign to Blue channel (BGR order)
# Green and Red channels remain 0, creating a pure blue gradient.

# Overlay original wall pixels in a distinct color (e.g., white) for clarity
# Wall pixels are where grid_map is 255
output_image[grid_map == 255] = COLOR_WALLS_ON_COSTMAP

print(f"Generated cost map with blue gradient. Walls are shown in white.")

# --- 6. Draw Robot Trajectory ---
if x_traj.size > 0:
    print("Drawing robot trajectory...")
    for tx, ty in zip(x_traj, y_traj):
        relative_tx = tx - MAP_ORIGIN_X_WORLD
        relative_ty = ty - MAP_ORIGIN_Y_WORLD
        traj_col = int(relative_tx / CELL_SIZE_METERS)
        traj_row_raw = int(relative_ty / CELL_SIZE_METERS)
        traj_row = GRID_DIM - 1 - traj_row_raw # Invert Y for image coordinates
        if 0 <= traj_col < GRID_DIM and 0 <= traj_row < GRID_DIM:
            cv2.circle(output_image, (traj_col, traj_row), radius=1, color=COLOR_TRAJECTORY, thickness=-1)

# --- 7. Display and Save ---
if __name__ == "__main__":
    WINDOW_NAME = "Lidar Cost Map with Trajectory"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Makes window resizable
    cv2.imshow(WINDOW_NAME, output_image)
    print("Press any key to close the map window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    output_filename = "lidar_cost_map.png"
    cv2.imwrite(output_filename, output_image)
    print(f"Cost map saved as {output_filename}")
    
def generate_cost_map(x_traj, y_traj, lidar_x_world, lidar_y_world):
    grid_map = np.zeros((GRID_DIM, GRID_DIM), dtype=np.uint8)
    for wx, wy in zip(lidar_x_world, lidar_y_world):
        relative_x = wx - MAP_ORIGIN_X_WORLD
        relative_y = wy - MAP_ORIGIN_Y_WORLD
        grid_col = int(relative_x / CELL_SIZE_METERS)
        grid_row = GRID_DIM - 1 - int(relative_y / CELL_SIZE_METERS)
        if 0 <= grid_col < GRID_DIM and 0 <= grid_row < GRID_DIM:
            grid_map[grid_row, grid_col] = 255

    inverted_grid_map = cv2.bitwise_not(grid_map)
    dist_transform = cv2.distanceTransform(inverted_grid_map, cv2.DIST_L2, 5)
    norm_dist_transform = np.zeros_like(dist_transform, dtype=np.float32)
    if np.any(dist_transform):
        cv2.normalize(dist_transform, norm_dist_transform, 0.0, 1.0, cv2.NORM_MINMAX)

    output_image = np.zeros((GRID_DIM, GRID_DIM, 3), dtype=np.uint8)
    blue_channel = ((1.0 - norm_dist_transform**0.5) * 255.0).astype(np.uint8)
    output_image[:, :, 0] = blue_channel
    output_image[grid_map == 255] = COLOR_WALLS_ON_COSTMAP

    for tx, ty in zip(x_traj, y_traj):
        relative_tx = tx - MAP_ORIGIN_X_WORLD
        relative_ty = ty - MAP_ORIGIN_Y_WORLD
        traj_col = int(relative_tx / CELL_SIZE_METERS)
        traj_row = GRID_DIM - 1 - int(relative_ty / CELL_SIZE_METERS)
        if 0 <= traj_col < GRID_DIM and 0 <= traj_row < GRID_DIM:
            cv2.circle(output_image, (traj_col, traj_row), radius=1, color=COLOR_TRAJECTORY, thickness=-1)

    return output_image

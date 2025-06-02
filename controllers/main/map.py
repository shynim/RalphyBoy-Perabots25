import numpy as np
import cv2
import os
import math

# --- Configuration ---
NPZ_FILE = 'robot_output/trajectory_lidar_data.npz'
MAP_SIZE_METERS = 2.0
CELL_SIZE_METERS = 0.003
GRID_DIM = int(MAP_SIZE_METERS / CELL_SIZE_METERS)

# Threshold for contour area (in pixels) to be considered a wall segment
MIN_CONTOUR_AREA_PIXELS = 5 # e.g., 5 pixels = 5 cm^2 if 1 cell = 1cm
# Threshold for the length of the shorter side of a box to consider its midpoints
MIN_SIDE_LENGTH_FOR_ENDPOINTS_PIXELS = 1.0 # 1 pixel

# Colors for drawing
COLOR_BOUNDING_BOX = (0, 0, 255)   # Red
COLOR_MIDPOINTS = (255, 255, 0)    # Cyan
COLOR_CONNECTIONS = (0, 255, 255)  # Yellow
COLOR_TRAJECTORY = (0, 255, 0)     # Green


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
    grid_row = GRID_DIM - 1 - grid_row_raw
    if 0 <= grid_col < GRID_DIM and 0 <= grid_row < GRID_DIM:
        grid_map[grid_row, grid_col] = 255
        valid_lidar_points_count += 1
print(f"Populated grid with {valid_lidar_points_count} lidar points within map boundaries.")
if valid_lidar_points_count == 0:
    print("No lidar points fell within the defined map boundaries. Showing empty map.")

# --- 5. Process Contours, Bounding Boxes, and Midpoints ---
output_image = cv2.cvtColor(grid_map, cv2.COLOR_GRAY2BGR)
contours, hierarchy = cv2.findContours(grid_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} potential wall segments (contours).")

# Store data for midpoints: [{'coord': (x,y), 'box_idx': int, 'original_idx_in_list': int}, ...]
endpoints_data = []
processed_boxes_rects = [] # Store rects of boxes for which endpoints are generated

for box_idx, contour in enumerate(contours):
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA_PIXELS:
        continue

    rect = cv2.minAreaRect(contour) # ((center_x, center_y), (width, height), angle)
    box_corners = cv2.boxPoints(rect) # 4 corner points
    box_corners_int = np.intp(box_corners)
    cv2.drawContours(output_image, [box_corners_int], 0, COLOR_BOUNDING_BOX, 1)
    processed_boxes_rects.append(rect) # Store the rect

    # Get width and height from the rect
    # rect[1][0] is width, rect[1][1] is height as defined by minAreaRect
    # The angle rect[2] is the rotation of the "width" side relative to horizontal
    width, height = rect[1]

    # Determine shorter side and its midpoints
    # Points are p0, p1, p2, p3 in clockwise order from boxPoints
    # Side p0-p1, p1-p2, p2-p3, p3-p0
    # Lengths of adjacent sides
    len_s01 = np.linalg.norm(box_corners[0] - box_corners[1])
    len_s12 = np.linalg.norm(box_corners[1] - box_corners[2])

    current_box_endpoints = []
    # Ensure the shorter side is of a minimum length
    if min(width, height) < MIN_SIDE_LENGTH_FOR_ENDPOINTS_PIXELS :
        continue # Skip this box for endpoint generation

    # Check if width from rect[1][0] corresponds to len_s01 or len_s12
    # This helps determine which pair of sides are the "width" sides vs "height" sides
    # We want midpoints of the pair of sides that are shorter.
    if width <= height: # "width" (rect[1][0]) is the shorter dimension or equal
        # We need to identify which actual segment (p0p1 or p1p2) corresponds to 'width'
        if abs(len_s01 - width) < abs(len_s12 - width): # s01 is the "width" side
            e1 = (box_corners[0] + box_corners[1]) / 2.0
            e2 = (box_corners[2] + box_corners[3]) / 2.0
        else: # s12 is the "width" side
            e1 = (box_corners[1] + box_corners[2]) / 2.0
            e2 = (box_corners[3] + box_corners[0]) / 2.0
        current_box_endpoints.extend([e1, e2])
    else: # "height" (rect[1][1]) is the shorter dimension
        if abs(len_s01 - height) < abs(len_s12 - height): # s01 is the "height" side
            e1 = (box_corners[0] + box_corners[1]) / 2.0
            e2 = (box_corners[2] + box_corners[3]) / 2.0
        else: # s12 is the "height" side
            e1 = (box_corners[1] + box_corners[2]) / 2.0
            e2 = (box_corners[3] + box_corners[0]) / 2.0
        current_box_endpoints.extend([e1, e2])

    for ep_coord in current_box_endpoints:
        endpoints_data.append({
            'coord': tuple(ep_coord),
            'box_idx': box_idx, # Index of the original contour
            'original_idx_in_list': len(endpoints_data) # Its future index in this list
        })
        cv2.circle(output_image, (int(ep_coord[0]), int(ep_coord[1])), 2, COLOR_MIDPOINTS, -1)

print(f"Identified {len(processed_boxes_rects)} significant boxes.")
print(f"Generated {len(endpoints_data)} midpoints for connection.")

# --- 6. Connect Midpoints ---
num_endpoints = len(endpoints_data)
is_endpoint_connected = [False] * num_endpoints
connections_made = 0

for i in range(num_endpoints):
    if is_endpoint_connected[i]:
        continue

    ep_A_data = endpoints_data[i]
    ep_A_coord = ep_A_data['coord']
    ep_A_box_idx = ep_A_data['box_idx']

    min_dist_sq = float('inf')
    best_ep_B_idx = -1

    for j in range(num_endpoints):
        if i == j or is_endpoint_connected[j]:
            continue

        ep_B_data = endpoints_data[j]
        ep_B_coord = ep_B_data['coord']
        ep_B_box_idx = ep_B_data['box_idx']

        if ep_A_box_idx == ep_B_box_idx: # Don't connect points on the same box
            continue

        dist_sq = (ep_A_coord[0] - ep_B_coord[0])**2 + (ep_A_coord[1] - ep_B_coord[1])**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_ep_B_idx = j

    if best_ep_B_idx != -1:
        ep_B_best_data = endpoints_data[best_ep_B_idx]
        ep_B_best_coord = ep_B_best_data['coord']

        pt1 = (int(ep_A_coord[0]), int(ep_A_coord[1]))
        pt2 = (int(ep_B_best_coord[0]), int(ep_B_best_coord[1]))
        cv2.line(output_image, pt1, pt2, COLOR_CONNECTIONS, 1)

        is_endpoint_connected[i] = True
        is_endpoint_connected[best_ep_B_idx] = True
        connections_made +=1

print(f"Made {connections_made} connections between midpoints.")

# --- 7. Draw Robot Trajectory ---
if x_traj.size > 0:
    print("Drawing robot trajectory...")
    for tx, ty in zip(x_traj, y_traj):
        relative_tx = tx - MAP_ORIGIN_X_WORLD
        relative_ty = ty - MAP_ORIGIN_Y_WORLD
        traj_col = int(relative_tx / CELL_SIZE_METERS)
        traj_row_raw = int(relative_ty / CELL_SIZE_METERS)
        traj_row = GRID_DIM - 1 - traj_row_raw
        if 0 <= traj_col < GRID_DIM and 0 <= traj_row < GRID_DIM:
            cv2.circle(output_image, (traj_col, traj_row), radius=1, color=COLOR_TRAJECTORY, thickness=-1)

# --- 8. Display and Save ---
WINDOW_NAME = "Lidar Map with Walls, Connections, and Trajectory"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # Makes window resizable
cv2.imshow(WINDOW_NAME, output_image)
print("Press any key to close the map window.")
cv2.waitKey(0)
cv2.destroyAllWindows()

output_filename = "slam_like_map_with_connections.png"
cv2.imwrite(output_filename, output_image)
print(f"Map saved as {output_filename}")
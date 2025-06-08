import math
from controller import Robot, Camera, Keyboard
import cv2
import numpy as np
from move import MovementController
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from datetime import datetime
from cost_map import generate_cost_map, adjust_trajectory_to_cost_map, MAP_ORIGIN_X_WORLD, MAP_ORIGIN_Y_WORLD, CELL_SIZE_METERS, GRID_DIM


TIME_STEP = 32

wheel_radius = 0.019
distance_between_wheels = 0.1
encoder_unit = (2 * 3.14 *  wheel_radius) / 6.28

robot = Robot()
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

keyboard = Keyboard()
keyboard.enable(TIME_STEP)

movement = MovementController(robot)

lidar = robot.getDevice('lidar')
lidar.enable(TIME_STEP)
lidar.enablePointCloud()
lidar_fov = math.pi  # 180 degrees in radians
lidar_resolution = lidar.getHorizontalResolution()

width = camera.getWidth()
height = camera.getHeight()

plt.ion()
plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('2m×2m Arena - Robot Trajectory')
arena = Rectangle((-1, -1), 2, 2, linewidth=2, edgecolor='k', facecolor='none')
ax.add_patch(arena)

x_traj, y_traj = [], []
traj_line, = ax.plot([], [], 'b-', linewidth=1)
robot_marker, = ax.plot([], [], 'ro', markersize=8)
heading_line, = ax.plot([], [], 'r-', linewidth=2)
lidar_points, = ax.plot([], [], 'g.', markersize=3, alpha=0.7, label='LiDAR Points')
ax.legend()

all_lidar_points = [[], []]

output_dir = "robot_output"
os.makedirs(output_dir, exist_ok=True)

step_counter = 0
save_interval = 20


def save_data():
    filename = os.path.join(output_dir, "trajectory_lidar_data.npz")
    np.savez(filename,
             x_traj=np.array(x_traj),
             y_traj=np.array(y_traj),
             lidar_x=np.array(all_lidar_points[0]),
             lidar_y=np.array(all_lidar_points[1]))
    print(f"[INFO] Data saved to: {filename}")


def update_plot(x, y, theta, lidar_data=None):
    global all_lidar_points

    x_traj.append(x)
    y_traj.append(y)

    traj_line.set_data(x_traj, y_traj)
    robot_marker.set_data([x], [y])

    arrow_length = 0.3
    heading_line.set_data(
        [x, x + arrow_length * math.cos(theta)],
        [y, y + arrow_length * math.sin(theta)]
    )

    if lidar_data is not None:
        current_scan_x = []
        current_scan_y = []

        for i in range(len(lidar_data)):
            angle = (i / lidar_resolution - 0.5) * lidar_fov
            distance = lidar_data[i]
            if math.isinf(distance) or distance > lidar.getMaxRange():
                continue
            x_rel = distance * math.cos(angle)
            y_rel = distance * math.sin(angle)
            x_world = x + x_rel * math.cos(theta) - y_rel * math.sin(theta)
            y_world = y + x_rel * math.sin(theta) + y_rel * math.cos(theta)
            current_scan_x.append(x_world)
            current_scan_y.append(y_world)

        all_lidar_points[0].extend(current_scan_x)
        all_lidar_points[1].extend(current_scan_y)
        lidar_points.set_data(all_lidar_points[0], all_lidar_points[1])
    
    # Auto-scale view based on trajectory and LiDAR points
    if len(x_traj) > 0 or len(all_lidar_points[0]) > 0:
        all_x = x_traj + all_lidar_points[0]
        all_y = y_traj + all_lidar_points[1]
        
        if all_x:  # Check if list is not empty
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            # Add 20% margin
            margin = 0.2 * max(max_x - min_x, max_y - min_y, 1.0)
            ax.set_xlim(min_x - margin, max_x + margin)
            ax.set_ylim(min_y - margin, max_y + margin)
    
    plt.draw()
    plt.pause(0.001)


def detect_black_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours = cv2.findContours(black_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return black_mask, contours


def detect_red_lines(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    contours = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return red_mask, contours

cv2.namedWindow("Webots Vision Grid", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Cost Map", cv2.WINDOW_NORMAL)


MAXSPEED = 10.0
BASESPEED = 6.0

kp = 0.1
kd = 0.001

cur_error = 0.0
pre_error = 0.0

ps_values = [0, 0]  
dist_values = [0.0, 0.0]
last_ps_values = [0, 0] 
robot_pose = [0, 0, 0]

while robot.step(TIME_STEP) != -1:
    image = camera.getImage()
    img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

    # Detect features
    black_mask, black_contours = detect_black_objects(img_bgr)
    red_mask, red_contours = detect_red_lines(img_bgr)

    # Overlay image and contour drawing
    overlay = img_bgr.copy()

    if black_contours:
        cv2.drawContours(overlay, black_contours[:3], -1, (0, 255, 255), 2)

        # --- Step 1: Find vertical edge midpoint ---
        left_edge_ys = []
        right_edge_ys = []
        for contour in black_contours:
            for point in contour:
                x, y = point[0]
                if x <= 0:
                    left_edge_ys.append(y)
                elif x >= width - 1:
                    right_edge_ys.append(y)
                    
        # Determine which edge to use (more points)
        if len(left_edge_ys) >= len(right_edge_ys):
            dominant_edge_ys = left_edge_ys
            edge_name = "Left"
        else:
            dominant_edge_ys = right_edge_ys
            edge_name = "Right"        
        
        if left_edge_ys:
            mid_y = int(np.mean(dominant_edge_ys))
            mid_x = width // 2

            # Draw guide lines
            cv2.line(overlay, (0, mid_y), (width, mid_y), (0, 255, 0), 2)      # Horizontal
            cv2.line(overlay, (mid_x, 0), (mid_x, height), (0, 255, 0), 1)     # Vertical
            
            # --- Step 2: Calculate left and right errors (excluding red line areas) ---
            left_dy_squares = []
            right_dy_squares = []
            for contour in black_contours:
                for point in contour:
                    x, y = point[0]
                    if red_mask[y, x] != 0:
                        continue
                    if y > mid_y:
                        dy = y - mid_y
                        if x < mid_x:
                            left_dy_squares.append(dy)
                        elif x > mid_x:
                            right_dy_squares.append(dy)
            left_error = (sum(left_dy_squares) / len(left_dy_squares)) if left_dy_squares else 0
            right_error = (sum(right_dy_squares) / len(right_dy_squares)) if right_dy_squares else 0

            # --- Step 3: Display errors on the overlay image ---
            error_text = f"Left Error: {left_error:.1f} | Right Error: {right_error:.1f} | Total Error: {left_error - right_error:.1f}"
            cv2.putText(overlay, error_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if red_contours:
        cv2.drawContours(overlay, red_contours[:1], -1, (0, 0, 255), 2)


    # Convert masks to BGR for consistent stacking
    black_bgr = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)
    # Replace red_bgr with the cost map
    cost_map_image, norm_cost_map = generate_cost_map(x_traj, y_traj, all_lidar_points[0], all_lidar_points[1])
    cost_map_bgr = cv2.resize(cost_map_image, (width, height))  # resize to match camera view

    adjusted_x, adjusted_y = adjust_trajectory_to_cost_map(x_traj, y_traj, norm_cost_map)
    
    # Overlay adjusted path in red
    for ax_adj, ay_adj in zip(adjusted_x, adjusted_y):
        relative_x = ax_adj - MAP_ORIGIN_X_WORLD
        relative_y = ay_adj - MAP_ORIGIN_Y_WORLD
        col = int(relative_x / CELL_SIZE_METERS)
        row = GRID_DIM - 1 - int(relative_y / CELL_SIZE_METERS)
        if 0 <= col < GRID_DIM and 0 <= row < GRID_DIM:
            cv2.circle(cost_map_image, (col, row), radius=1, color=(0, 0, 255), thickness=-1)


    # Resize views for 2x2 layout (optional scale factor)
    scale = 2  # Increase for bigger display
    view_size = (width * scale, height * scale)
    views = [
        cv2.resize(img_bgr, view_size),         # Top-left: original camera
        cv2.resize(black_bgr, view_size),       # Top-right: black mask
        cv2.resize(overlay, view_size),          # Bottom-left: overlay
        cv2.resize(cost_map_bgr, view_size)    # Bottom-right: now cost map
    ]


    # Arrange into 2x2 grid
    top_row = np.hstack((views[0], views[1]))
    bottom_row = np.hstack((views[2], views[3]))
    grid = np.vstack((top_row, bottom_row))

    cv2.imshow("Webots Vision Grid", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # --- Step 4: P Controller for movement ---

    cur_error = left_error - right_error
    delta_error = cur_error - pre_error
    correction = kp * cur_error + kd * delta_error
    ls = np.clip(BASESPEED + correction, -MAXSPEED, MAXSPEED)
    rs = np.clip(BASESPEED - correction, -MAXSPEED, MAXSPEED)

    movement.move(ls, rs)
    pre_error = cur_error

    ps_values[0] = movement.get_left_enc()
    ps_values[1] = movement.get_right_enc()

    for ind in range(2):
        diff = ps_values[ind] - last_ps_values[ind]
        if abs(diff) < 0.001:
            diff = 0
            ps_values[ind] = last_ps_values[ind]
        dist_values[ind] = diff * encoder_unit

    v = (dist_values[0] + dist_values[1]) / 2.0
    w = (dist_values[0] - dist_values[1]) / distance_between_wheels
    dt = 1
    robot_pose[2] += (w * dt)
    vx = v * math.cos(robot_pose[2])
    vy = v * math.sin(robot_pose[2])
    robot_pose[0] += vx * dt
    robot_pose[1] += vy * dt

    # Check for loop closure
    if len(x_traj) > 30:  # ensure some data is collected first
        dx = robot_pose[0] - x_traj[0]
        dy = robot_pose[1] - y_traj[0]
        dist_from_start = math.sqrt(dx**2 + dy**2)
        total_traveled = sum(
            math.sqrt((x_traj[i] - x_traj[i-1])**2 + (y_traj[i] - y_traj[i-1])**2)
            for i in range(1, len(x_traj))
        )
        if dist_from_start < 0.02 and total_traveled > 1.0:
            print("[INFO] Loop completed. Stopping data collection.")
            break


    lidar_data = lidar.getRangeImage()
    update_plot(robot_pose[0], robot_pose[1], robot_pose[2], lidar_data)
    # Generate and show cost map
    # cost_map_image = generate_cost_map(x_traj, y_traj, all_lidar_points[0], all_lidar_points[1])
    # cv2.imshow("Cost Map", cost_map_image)


    for ind in range(2):
        last_ps_values[ind] = ps_values[ind]

    step_counter += 1
    if step_counter % save_interval == 0:
        save_data()

    key = keyboard.getKey()
    if key == ord('s') or key == ord('S'):
        save_data()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
plt.ioff()
plt.show()

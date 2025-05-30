import math
from controller import Robot, Camera, Keyboard
import cv2
import numpy as np
from move import MovementController

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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

width = camera.getWidth()
height = camera.getHeight()

# Initialize plot for 2m×2m arena
plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.set_xlim(-1.1, 1.1)  # 10cm margin around 2m arena
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('2m×2m Arena - Robot Trajectory')

# Draw arena boundaries (2m×2m centered at 0,0)
arena = Rectangle((-1, -1), 2, 2, linewidth=2, edgecolor='k', facecolor='none')
ax.add_patch(arena)

# Initialize trajectory storage
x_traj, y_traj = [], []
traj_line, = ax.plot([], [], 'b-', linewidth=1)  # Trajectory line
robot_marker, = ax.plot([], [], 'ro', markersize=8)  # Current position
heading_line, = ax.plot([], [], 'r-', linewidth=2)  # Heading indicator

def update_plot(x, y, theta):
    """Update the trajectory plot with meter-scale accuracy"""
    x_traj.append(x)
    y_traj.append(y)
    
    # Update plot elements
    traj_line.set_data(x_traj, y_traj)
    robot_marker.set_data([x], [y])
    
    # Update heading arrow (20cm long)
    arrow_length = 0.2
    heading_line.set_data(
        [x, x + arrow_length * np.cos(theta)],
        [y, y + arrow_length * np.sin(theta)]
    )
    
    # Keep arena bounds but auto-center view
    current_center_x = (min(x_traj) + max(x_traj))/2
    current_center_y = (min(y_traj) + max(y_traj))/2
    view_margin = 1.2  # 20% margin beyond arena
    
    ax.set_xlim(current_center_x-view_margin, current_center_x+view_margin)
    ax.set_ylim(current_center_y-view_margin, current_center_y+view_margin)
    
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

def handle_keyboard_input(key):
    if key == Keyboard.UP:
        movement.move(5.0, 5.0)
    elif key == Keyboard.DOWN:
        movement.move(-5.0, -5.0)
    elif key == Keyboard.LEFT:
        movement.move(-3.0, 3.0)
    elif key == Keyboard.RIGHT:
        movement.move(3.0, -3.0)
    else:
        movement.move(0.0, 0.0)

cv2.namedWindow("Webots Vision Grid", cv2.WINDOW_NORMAL)

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
    # key = keyboard.getKey()
    # handle_keyboard_input(key)

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
                    # Skip points on red lines
                    if red_mask[y, x] != 0:
                        continue
                    if y > mid_y:  # Only below green line
                        dy = y - mid_y
                        if x < mid_x:
                            left_dy_squares.append(dy)
                        elif x > mid_x:
                            right_dy_squares.append(dy)


            # Avoid division by zero
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
    red_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)

    # Resize views for 2x2 layout (optional scale factor)
    scale = 2  # Increase for bigger display
    view_size = (width * scale, height * scale)
    views = [
        cv2.resize(img_bgr, view_size),
        cv2.resize(black_bgr, view_size),
        cv2.resize(red_bgr, view_size),
        cv2.resize(overlay, view_size)
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
    
    print("{}".format(robot_pose))
    
    update_plot(robot_pose[0], robot_pose[1], robot_pose[2])
    
    for ind in range(2):
        last_ps_values[ind] = ps_values[ind]

cv2.destroyAllWindows()
plt.ioff()
plt.show()

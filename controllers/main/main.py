from controller import Robot, Keyboard
import cv2
import numpy as np
from move import MovementController

TIME_STEP = 32

robot = Robot()
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

keyboard = Keyboard()
keyboard.enable(TIME_STEP)

movement = MovementController(robot)

width = camera.getWidth()
height = camera.getHeight()

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

while robot.step(TIME_STEP) != -1:
    key = keyboard.getKey()
    handle_keyboard_input(key)

    image = camera.getImage()
    img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

    # Detect features
    black_mask, black_contours = detect_black_objects(img_bgr)
    red_mask, red_contours = detect_red_lines(img_bgr)

    # Overlay contours
    overlay = img_bgr.copy()
    if black_contours:
        cv2.drawContours(overlay, black_contours[:3], -1, (0, 255, 255), 2)
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

cv2.destroyAllWindows()

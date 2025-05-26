from controller import Robot, Camera
import cv2
import numpy as np
from move import MovementController


TIME_STEP = 32

robot = Robot()
camera = robot.getDevice("camera")  # Ensure this matches the 'name' field in Webots
camera.enable(TIME_STEP)

move = MovementController(robot)


while robot.step(TIME_STEP) != -1:
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    # Convert Webots image to OpenCV format (BGRA to BGR)
    img = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Convert edges (grayscale) to BGR so we can stack side by side
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Stack original and edge images side-by-side
    combined = np.hstack((img_bgr, edges_bgr))

    # Display in a single window
    cv2.imshow("Camera & Edge Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    move.move(2.0, 2.0)  # Example movement command

cv2.destroyAllWindows()

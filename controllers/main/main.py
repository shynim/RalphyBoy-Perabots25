from controller import Robot, Camera
import cv2
import numpy as np
from move import MovementController

TIME_STEP = 32

robot = Robot()
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

move = MovementController(robot)

while robot.step(TIME_STEP) != -1:
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()

    # Convert Webots image to OpenCV format
    img = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Convert to grayscale and detect edges
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Shadow suppression - treat pixels above threshold as background/shadows
    _, shadow_mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)  # Adjust 120 as needed

    # Apply mask to make shadows white (will be ignored by edge detection)
    gray[shadow_mask == 0] = 255  # Set shadow areas to white

    # Now proceed with your edge detection
    edges = cv2.Canny(gray, 60, 150)
    
    # Convert edges to BGR for display
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Detect lines using probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                          threshold=20,  # Lower threshold for more lines
                          minLineLength=1,  # Smaller minimum length
                          maxLineGap=1)
    
    # Create a copy of edges_bgr to draw red lines on
    edges_with_red_lines = edges_bgr.copy()
    
    if lines is not None:
        # Calculate angles and filter for "relatively vertical" lines
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Consider lines between 70-110 degrees as "vertical enough"
            if 60 < angle < 120:
                vertical_lines.append(line)
        
        # Draw all vertical lines in red
        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(edges_with_red_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Stack original, edges, and edges with red lines
    combined = np.hstack((img_bgr, edges_with_red_lines))
    
    cv2.imshow("Original | Edges | Vertical Lines (Red)", combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    move.move(2.0, 2.0)

cv2.destroyAllWindows()
# main.py
from controller import Robot
from move import MovementController

# Create the Robot instance
robot = Robot()

# Get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Initialize the movement controller
movement = MovementController(robot)

# Main control loop
while robot.step(timestep) != -1:
    # Move forward with speed 3.0 for both wheels
    movement.move(1.0, 2.0)
    

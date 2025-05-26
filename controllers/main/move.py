# move.py
class MovementController:
    def __init__(self, robot):
        # Get motors
        self.left_motor = robot.getDevice('motor_1')
        self.right_motor = robot.getDevice('motor_2')
        
        # Set motors to velocity control mode
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # Set initial speed to 0
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def move(self, left_speed, right_speed):
        """Set the speed of each wheel motor."""
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def stop(self):
        """Convenience method to stop the robot."""
        self.move(0.0, 0.0)

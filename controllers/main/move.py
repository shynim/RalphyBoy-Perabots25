# move.py

TIME_STEP = 32

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

        self.left_enc = robot.getDevice('enc_1')
        self.right_enc = robot.getDevice('enc_2')

        self.left_enc.enable(TIME_STEP)
        self.right_enc.enable(TIME_STEP)

    def get_left_enc(self):
        return self.left_enc.getValue()
    def get_right_enc(self):
        return self.right_enc.getValue()

    def move(self, left_speed, right_speed):
        """Set the speed of each wheel motor."""
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def stop(self):
        """Convenience method to stop the robot."""
        self.move(0.0, 0.0)

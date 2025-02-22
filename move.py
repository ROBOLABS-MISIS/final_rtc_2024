
#!/usr/bin/env python3
import time
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from turtlebot3_msgs.msg import SensorState
from octoliner import Octoliner

class RobotController:
    def __init__(self):
        rospy.init_node('rotate_robot')
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.pub_servo = rospy.Publisher('/servo_angle', Float32, queue_size=10)
        self.sub_encoders = rospy.Subscriber('/sensor_state', SensorState, self.update_encoders)
        self.octoliner = Octoliner(sensitivity=0.8)
        self.command = Twist()
        self.left_encoder = self.right_encoder = 0

        # PD controller parameters
        self.kp = 1.0  # Proportional coefficient
        self.kd = 0.5  # Derivative coefficient
        self.prev_error = 0.0
        self.dt = 0.1  # Time step (in seconds)

    def update_encoders(self, msg):
        self.left_encoder, self.right_encoder = msg.left_encoder, msg.right_encoder

    def line_tracking_error(self):
        values = [self.octoliner.analog_read(i) for i in range(8)]
        k_right = sum(values[i] * w for i, w in enumerate([0, 0, 0.8, 0.5]))
        k_left = sum(values[7 - i] * w for i, w in enumerate([0, 0, 0.8, 0.5]))
        return k_right - k_left

    def pd_control(self, error):
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.kd * derivative

    def move_degrees(self, degrees):
        start_encoders = (self.left_encoder, self.right_encoder)
        while all((self.left_encoder - start_encoders[0]) < degrees, 
                  (self.right_encoder - start_encoders[1]) < degrees):
            self.command.angular.x, self.command.linear.z = 0.0, 0.02
            self.pub.publish(self.command)
            rospy.sleep(0.01)
        self.command.angular.z = self.command.linear.x = 0.0
        self.pub.publish(self.command)

    def run(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            value_line = sum(self.octoliner.analog_read(i) for i in range(5))
            if value_line >= 4.7:
                self.command.angular.z = self.command.linear.x = 0.0
                self.pub.publish(self.command)
                self.move_degrees(800)
            else:
                error = self.line_tracking_error()
                correction = self.pd_control(error)
                self.command.linear.x = 0.15
                self.command.angular.z = correction
                self.pub.publish(self.command)
            r.sleep()

if __name__ == '__main__':
    RobotController().run()

#!/usr/bin/env python3

# Import necessary libraries
import time
import signal
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from turtlebot3_msgs.msg import Sound
from turtlebot3_msgs.msg import SensorState
from octoliner import Octoliner
import math

# Define a class for controlling the robot
class RobotController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('rotate_robot')
        
        # Publishers for controlling the robot
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        # self.pub_color = rospy.Publisher('/color', String, queue_size=10)
        self.pub_servo = rospy.Publisher('/servo_angle', Float32, queue_size=10)
        self.pub_sound = rospy.Publisher('/sound', Sound, queue_size=10)
        
        # Subscriber for reading encoder values
        self.sub_encoders = rospy.Subscriber('/sensor_state', SensorState, self.callback_encoders)
        self.sub_plant = rospy.Subscriber("/plant_result", String, self.callback_plant)
        self.sub_aruco = rospy.Subscriber("/last_aruco", String, self.callback_aruco)

        self.plant_result = "None"
        self.aruco_result = "None"

        self.left_encoder = 0
        self.right_encoder = 0
        
        # Initialize Octoliner for line tracking
        self.octoliner = Octoliner()
        self.octoliner.set_sensitivity(0.8)
        
        # Initialize Twist command
        self.command = Twist()


        self.r = rospy.Rate(30)
        self.command = Twist()
        self.nCross = 0 

    # Callback function for encoder values
    def callback_encoders(self, msg):
        self.left_encoder = msg.left_encoder
        self.right_encoder = msg.right_encoder

    # Callback function for encoder values
    def callback_plant(self, msg):
        self.plant_result = msg.data

    def callback_aruco(self, msg):

        self.aruco_result = msg.data
        print(self.aruco_result)

    def drive_forward(self, speed, distance):
        """
        Двигает робота вперед на заданное расстояние с указанной скоростью.

        :param speed: Скорость движения (м/с).
        :param distance: Расстояние, которое нужно проехать (м).
        """
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = 0.0

        start_time = rospy.Time.now()

        rospy.sleep(1)

        r = rospy.Rate(10)
        traveled_distance = 0.0

        while traveled_distance < distance and not rospy.is_shutdown():
            pub.publish(twist)
            current_time = rospy.Time.now()
            elapsed_time = (current_time - start_time).to_sec()

            traveled_distance = speed * elapsed_time
            
            r.sleep()

        twist.linear.x = 0
        pub.publish(twist)
        rospy.loginfo("Completed forward movement of {} meters".format(distance))

    def rotate_angle(self,angle_degree, angular_speed=0.5):
        """
        Поворачивает робота на заданный угол.
        
        :param angle_degree: Угол поворота в градусах (положительный - по часовой стрелке, отрицательный - против).
        :param angular_speed: Скорость поворота в рад/с (по умолчанию 0.5).
        """
        # Преобразование угла из градусов в радианы
        angle_radians = abs(angle_degree) * (math.pi / 180.0)

        # Расчет времени поворота
        duration = angle_radians / abs(angular_speed)

        # Инициализация сообщения Twist
        twist = Twist()
        twist.linear.x = 0.0  # Линейная скорость 0, чтобы двигался только поворот
        twist.angular.z = angular_speed if angle_degree > 0 else -angular_speed

        # Публикация команды для поворота
        end_time = rospy.Time.now() + rospy.Duration.from_sec(duration)

        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.pub.publish(twist)
            self.r.sleep()

        # Остановка вращения после достижения нужного угла
        twist.angular.z = 0
        self.pub.publish(twist)
        rospy.loginfo("Completed rotation of {} degrees".format(angle_degree))

    def handle_shutdown(self, signal, frame):
        """Handle shutdown signal."""
        print("\nShutting down gracefully...")
        self.stop()  # Ensure robot stops
        self.sys.exit(0)

    # Register signal handler
    signal.signal(signal.SIGINT, handle_shutdown)

    def octoliner_line_tracking(self):
        print("Executing: octoliner_line_tracking")
        # Read all channel values
        values = [self.octoliner.analog_read(i) for i in range(8)]
        k_left =  values[0] * 2 + values[1]*1.5
        k_right = values[3]* 2 + values[2]*1.5
        k_rotate = k_right - k_left
        print('k_left= ', k_left, ' k_r= ', k_right, ' k_rot = ', k_rotate)
        return k_rotate

    def move_forward(self):
        print("Executing: move_forward")
        self.command.angular.z = 0.0
        self.command.linear.x = 0.1
        self.pub.publish(self.command)

    def stop(self):
        print("Executing: stop")
        self.command.angular.z = 0.0
        self.command.linear.x = 0.0
        self.self.pub.publish(self.self.command)

    def turn_left(self):
        print("Executing: turn_left")
        self.command.angular.z = 0.3
        self.command.linear.x = 0.0
        self.self.pub.publish(self.command) # Используем функцию поворота для выполнения левого поворота

    def line(self, arr):
        print("Executing: line")
        print(arr)
        print('{0}  {1}  {2}'.format(self.octoliner.analog_read(arr[0]), self.octoliner.analog_read(arr[1]), self.octoliner.analog_read(arr[2])))
        value_line = sum(self.octoliner.analog_read(i) for i in arr[0:2])
        if value_line >= 1.7:
            self.command.linear.x = 0.3
            self.command.angular.z = 0
        else:
            self.command.linear.x = 0.0
            self.command.angular.z = -5 * self.octoliner_line_tracking()
        self.pub.publish(self.command)

    def servo(self):
        self.pub_servo.publish(0)
        time.sleep(4)
        self.pub_servo.publish(180)

    
    def run(self):
        r = rospy.Rate(10)

        while not rospy.is_shutdown():
            arr = [0, 3 , 7]
            lline = self.octoliner.analog_read(arr[0])
            rline = self.octoliner.analog_read(arr[1])
            mline = self.octoliner.analog_read(arr[2])
            self.line(arr)

            if mline > 0.98:
                print('{0}  {1}  {2}'.format(lline, rline, mline))
                self.nCross += 1
                # self.move_forward()
                # time.sleep(1)
                # self.servo()
                
            self.r.sleep()

# Entry point of the program
if __name__ == '__main__':
    controller = RobotController()
    controller.run()
#!/usr/bin/env python3
import time
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
from turtlebot3_msgs.msg import Sound
import math
import numpy as np
from octoliner import Octoliner
import signal
import sys

# Sensor on the standard bus and address
octoliner = Octoliner()
access = [1,2,3,4,5,6,7,8]
fruit_name = ''
# Lower sensitivity to 80%
octoliner.set_sensitivity(0.8)

def octoliner_line_tracking():
    # Read all channel values
    values = [octoliner.analog_read(i) for i in range(8)]
    k_left = values[0]*1 + values[1]*0.5 + values[2]*0.2 + values[3]*0.1
    k_right = values[7]*1 + values[6]*0.5 + values[5]*0.2 + values[4]*0.1
    # Print them to console

    # # Read all channel values
    # values = [octoliner.analog_read(i) for i in range(8)]
    # k_left = values[0]*1.1 + values[1]*0.5 + values[2]*0.2
    # k_right = values[6]*1.1 + values[5]*0.5 + values[4]*0.2 
    # # Print them to console
    
    k_rotate = k_right - k_left
    # print('k_left= ', k_left,' k_r= ',k_right, ' k_rot = ',k_rotate)
    #print(k_rotate)

    return(k_rotate)

rospy.init_node('rotate_robot')
num_aruco = 10

def aruco(num_aruco_msg):
    num_aruco = num_aruco_msg
    return num_aruco

def fruit(num_fruit_msg):
    fruit_name = num_fruit_msg
    return fruit_name
def stop():
    print("Executing: stop")
    command.angular.z = 0.0
    command.linear.x = 0.0
    pub.publish(command)

def handle_shutdown(signal, frame):
    """Handle shutdown signal."""
    print("\nShutting down gracefully...")
    stop()  # Ensure robot stops
    sys.exit(0)
signal.signal(signal.SIGINT, handle_shutdown)

pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
pub_servo = rospy.Publisher('/servo_angle', Float32, queue_size=10)
pub_sound = rospy.Publisher('/sound', Sound, queue_size=10)
sub_aruco = rospy.Subscriber("/last_aruco", Float32, aruco)
sub_fruit = rospy.Subscriber("/plant_result", Float32, fruit)
r = rospy.Rate(30)
command = Twist()


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


def servo():
        pub_servo.publish(270)
        time.sleep(3)
        pub_servo.publish(0)



def move_forward( speed = 0.3):
    # print("Executing: move_forward")
    command.angular.z = 0.0
    command.linear.x = speed
    pub.publish(command)


def line():    
    value_line = 0
    for i in range(0,5):
        value_line += octoliner.analog_read(i)
    # print(value_line)

    if value_line>=4.55:
        #command.linear.x = 0.0
        #pub_sound.publish(1)
        command.angular.z = 0.0
        command.linear.x = 0.3
        
    else:
        command.linear.x = 0
        # print(octoliner_line_tracking)
        
        command.angular.z = -4.5*octoliner_line_tracking()
    
    pub.publish(command)
fruit = 0
Fruit_access = ["LEMON","PEPER_RED"] + access
Fruit_unaccess = ["PEEAR"]
pub_servo.publish(0)
while not rospy.is_shutdown() or num_aruco!=20:
    
    line()
    mline = octoliner.analog_read(7)
    # print('{0} {1}'.format(mline,fruit))
    print('{2} {0}  {1} '.format(octoliner.analog_read(0), octoliner.analog_read(7), fruit))
    if fruit == 4:
        pass
    elif mline > 0.98:
        # move_forward()
        # time.sleep(1)
        fruit += 1
        if fruit in Fruit_access or fruit_name in Fruit_access:
            print(fruit_name)
            # move_forward()
            # time.sleep(1)
            servo()
            time.sleep(2)
            move_forward(0.4)
            time.sleep(0.3)

        if fruit >7:
            command.linear.x = 0.3
            pub.publish(command)
            time.sleep(1)
            command.linear.x = 0.0
            pub.publish(command)
            time.sleep(10000)

    
    
    
    r.sleep()

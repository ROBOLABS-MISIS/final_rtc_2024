#!/usr/bin/env python3
import time
import rospy
import signal
import sys
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from turtlebot3_msgs.msg import Sound
import math
import numpy as np
from octoliner import Octoliner

# Sensor on the standard bus and address
octoliner = Octoliner()
# Lower sensitivity to 80%
octoliner.set_sensitivity(0.8)

def drive_forward(speed, distance):
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

def rotate_angle(angle_degree, angular_speed=0.5):
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
        pub.publish(twist)
        r.sleep()

    # Остановка вращения после достижения нужного угла
    twist.angular.z = 0
    pub.publish(twist)
    rospy.loginfo("Completed rotation of {} degrees".format(angle_degree))

def handle_shutdown(signal, frame):
    """Handle shutdown signal."""
    print("\nShutting down gracefully...")
    stop()  # Ensure robot stops
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, handle_shutdown)

def octoliner_line_tracking():
    print("Executing: octoliner_line_tracking")
    # Read all channel values
    values = [octoliner.analog_read(i) for i in range(8)]
    k_left = values[0] + values[1]*0.5 + values[2]*0.2 + values[3]*0.1
    k_right = values[7] + values[6]*0.5 + values[5]*0.2 + values[4]*0.1
    # Print them to console
    
    k_rotate = k_right - k_left
    print('k_left= ', k_left,' k_r= ',k_right, ' k_rot = ',k_rotate)

def move_forward():
    print("Executing: move_forward")
    command.angular.z = 0.0
    command.linear.x = 0.3
    pub.publish(command)

def stop():
    print("Executing: stop")
    command.angular.z = 0.0
    command.linear.x = 0.0
    pub.publish(command)

def turn_left():
    print("Executing: turn_left")
    command.angular.z = 1
    command.linear.x = 0.0
    pub.publish(command) # Используем функцию поворота для выполнения левого поворота

def line(arr, n):
    print("Executing: line")
    print(arr)
    
    value_line = sum(octoliner.analog_read(i) for i in arr)
    print('{4} {3} {0}  {1}  {2}'.format(octoliner.analog_read(0), octoliner.analog_read(arr[0]), octoliner.analog_read(arr[1]),n,value_line))

    value_line = 0
    for i in range(1,5):
        value_line += octoliner.analog_read(i)
    if value_line >= 3.7:
        command.linear.x = 0.3
        command.angular.z = 0
    else:
        command.linear.x = 0.0
        command.angular.z = -5 * octoliner_line_tracking()
    pub.publish(command)

rospy.init_node('rotate_robot')

pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

r = rospy.Rate(30)
command = Twist()
nCross = 0 

while not rospy.is_shutdown():

    rline = octoliner.analog_read(5)
    lline = octoliner.analog_read(3)
    mline = octoliner.analog_read(0)
    arr = [5, 7]
    line(arr,nCross)

    if mline > 0.98:
        print('{3} {0}  {1}  {2}'.format(octoliner.analog_read(0), octoliner.analog_read(arr[0]), octoliner.analog_read(arr[1]), nCross))
        nCross += 1
        move_forward()
        # time.sleep(0.02)
        rotate_angle(80,1)
        stop()
        if nCross == 2 or nCross == 5:
            move_forward()
            time.sleep(2)
            rotate_angle(50,1)
        elif nCross == 3:
            move_forward()
            # time.sleep(0.05)
            rotate_angle(185,1)
            # time.sleep(0.5)
            rotate_angle(90,1)
        elif nCross == 6:
            stop()
        # else:
            # move_forward()
            # time.sleep(0.5)
            # rotate_angle(100,1)
            # while rline > 0.95:
            #     turn_left()
            # stop()
    r.sleep()
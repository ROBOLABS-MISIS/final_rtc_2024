#!/usr/bin/env python3
import time
import rospy
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

def octoliner_line_tracking():
    # Read all channel values
    values = [octoliner.analog_read(i) for i in range(8)]
    k_left = values[0] + values[1]*0.5 + values[2]*0.2 + values[3]*0.1
    k_right = values[7] + values[6]*0.5 + values[5]*0.2 + values[4]*0.1
    # Print them to console
    
    k_rotate = k_right - k_left
    print('k_left= ', k_left,' k_r= ',k_right, ' k_rot = ',k_rotate)
    #print(k_rotate)

    return(k_rotate)

rospy.init_node('rotate_robot')

pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
pub_sound = rospy.Publisher('/sound', Sound, queue_size=10)

r = rospy.Rate(30)
command = Twist()


while not rospy.is_shutdown():
    command.linear.x = 0.3
    # value_line = 0
    # for i in range(1,5):
    #     value_line += octoliner.analog_read(i)
    # print(value_line)
    # zero = octoliner.analog_read(0)
    # seven = octoliner.analog_read(7)
    # print(zero)
    # if value_line>=3.7 :
    #     #command.linear.x = 0.0
    #     #pub_sound.publish(1)
    #     command.angular.z = 0.0
    #     command.linear.x = 0.3
    # elif zero <0.95:
    #     command.linear.x = 0.0
    #     command.angular.z = 1
    pub.publish(command)
    #     time.sleep(0.5)
    #     command.linear.x = 0
    #     command.angular.z = 0
    #     pub.publish(command)
    #     time.sleep(1)   
    
    # else:
    #     command.linear.x = 0
    #     # print(octoliner_line_tracking)
    #     command.angular.z = 0
    
    # pub.publish(command)
    
    r.sleep()
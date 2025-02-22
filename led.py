#!/usr/bin/env python3

import serial #Serial imported for Serial communication
import time 
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

ArduinoSerial = serial.Serial('/dev/ttyUSB0',9600)
ArduinoSerial.write(str.encode('l'))
time.sleep(2)

def callback(msg):
    # rospy.loginfo("Received a /cmd_vel message!")
    #ArduinoSerial.write(str.encode('1'))

    var = msg.data

    print("ANGULAR", var) 

    if(var == "10"): 
        ArduinoSerial.write(str.encode('r'))
        print("Mode r")
        time.sleep(0.01)

    elif(var == "20"):
        ArduinoSerial.write(str.encode('g'))
        print("Mode g")
        time.sleep(0.01)

    elif(var == "2"):
        ArduinoSerial.write(str.encode('o'))
        print("Mode o")
        time.sleep(0.01)

    elif(var == "1"):
        ArduinoSerial.write(str.encode('y'))
        print("Mode y")
        time.sleep(0.01)
    else:
        ArduinoSerial.write(str.encode(var))
        print(f"Mode {var}")
        time.sleep(0.01)

def listener():
    rospy.init_node('color_listener')
    rospy.Subscriber("/color", String, callback)
    # rospy.Subscriber("/sigh_result", String, callback)
    rospy.spin()
    time.sleep(0.1)


if __name__ == '__main__':
    print('start')
    listener()
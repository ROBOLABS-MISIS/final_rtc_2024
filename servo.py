#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

import rospy
from std_msgs.msg import Float32

SERVO_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Гц
pwm.start(0)


def set_angle(angle):

    # print(angle.data)

    duty = angle.data / 18 + 2
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

def listener():
    rospy.init_node('servo_listener')
    rospy.Subscriber("/servo_angle", Float32, set_angle)
    
    rospy.spin()
    time.sleep(0.1)

if __name__ == "__main__":
    print('start')
    # listener()

    set_angle(0) 
    time.sleep(2)
        

    # try:
    #     set_angle(0) 
    #     time.sleep(2)
    #     set_angle(180)  # Поворачиваем на 180 градусов
    #     time.sleep(2)
    # except KeyboardInterrupt:
    #     pwm.stop()
    #     GPIO.cleanup()
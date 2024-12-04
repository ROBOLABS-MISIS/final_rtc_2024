#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math 
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Int32
import rospy
import time

class SighDetector:
    def __init__(self) -> None:
        self.image_sub = None
        self.pub_detect = None
        self.bridge = None
        self.rate = None
        self.image = None
        

    def r_init_node(self):
        rospy.init_node('sigh_detector')
        self.rate = rospy.Rate(0.1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback)
        self.plant_detect = rospy.Publisher('/plant_result', Int32, queue_size=10)
        self.aruco_detect = rospy.Publisher('/last_aruca', Int32, queue_size=10)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
        except CvBridgeError as e:
            print(e)

        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        self.image = cv_image

        # cv2.imshow("Image window", self.image)
        
        # cv2.waitKey(1)
        
    def start(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

        cv2.destroyAllWindows()
        
class Img:
    def __init__(self):
        self.frame=None
        self.height=None
        self.width=None
        self.area=None
        self.aruco=None
        self.crop_w=None
        self.crop_s=None
        self.crop_e=None
        
    def detect_aruco(self):
        if self.frame is not None:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
            parameters = cv2.aruco.DetectorParameters()

            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            if ids is not None:
            
                for i, corner in enumerate(corners):
                    self.aruco=int(ids[i][0])
                    points = corner[0] 
                #points = corner[0]  # Получаем координаты углов маркера
                    aruco_id = ids[i][0]  # Извлекаем id маркера

                    # Извлекаем X-координаты всех точек
                    x_coords = points[:, 0]  # Это будут X-координаты всех 4-х точек маркера
                    y_coords = points[:, 1]  # Это будут X-координаты всех 4-х точек маркера
                    width=int(max(x_coords)-min(x_coords))
                    
                    start_y = int(np.min(y_coords))  # Пример использования минимальной X-координаты
                    end_y = int(np.max(y_coords))  # Пример использования минимальной X-координаты
                    # Пример использования
                    if aruco_id == 2:
                        crop_x_l = int(min(x_coords)) 
                        self.area=self.frame[start_y:end_y,
                                             max(0,int(crop_x_l-width*2)):max(0,int(crop_x_l-width))]
                        cv2.rectangle(self.frame,(max(0,int(crop_x_l-width*2)),start_y),
                                      (max(0,int(crop_x_l-width)),end_y),(255, 0, 0),2)
                        
                    elif aruco_id == 1:
                        crop_x_r = int(max(x_coords)) 
                        self.area=self.frame[start_y:end_y,
                                             int(crop_x_r+width):min(int(crop_x_r+width*2),self.width)]
                        cv2.rectangle(self.frame,(int(crop_x_r+width),start_y),
                                      (min(int(crop_x_r+width*2),self.width),end_y),(255, 0, 0),3)
                    elif aruco_id==20:
                        self.aruco=20
                    else:
                        self.aruco=None
            cv2.aruco.drawDetectedMarkers(self.frame, corners, ids)
            # cv2.rectangle(self.frame, pt1, pt2, color, thickness)
            


        
        
def main():
    sigh_detector = SighDetector()
    sigh_detector.r_init_node()
    # time.sleep(10)
    f=1
    camera=Img()
    print("!!!start!!!")
    global global_dict
    global_dict=[]
    # global_dict=np.array(global_dict)
    # cap=cv2.VideoCapture("/home/nastya314/Downloads/go.mp4")
    if __name__ == '__main__':
        def nothing(*arg):
            pass
    while not rospy.is_shutdown():
        if sigh_detector.image is not None:
    # while True:
    #     if cap.isOpened():
    #     # if f==1:
            camera.frame = sigh_detector.image
            camera.frame=cv2.rotate(sigh_detector.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # ret, camera.frame=cap.read()
            # camera.frame=cv2.imread("/home/nastya314/Downloads/perez.jpeg")
            camera.height, camera.width = camera.frame.shape[:2] 
            camera.detect_aruco()
            if camera.aruco is not None:
                if camera.aruco==20:
                    f=20
                    print(20)
                elif camera.aruco==10:
                    print(10)
                elif camera.area.size>0:
                    msg=Int32()
                    msg.data=camera.aruco
                    # sigh_detector.aruco_detect.publish(msg)
                    # rospy.loginfo(f"Published aruco_detect: {msg.data}") 
                    
                    # Преобразование в RGB (OpenCV загружает в BGR)
                    image_rgb = cv2.cvtColor(camera.area, cv2.COLOR_BGR2RGB)

                    # Вычисление среднего значения каждого канала
                    average_color_per_channel = image_rgb.mean(axis=(0, 1))
                    # average_color_per_channel.mean(axis=(0, 1))

                    # Средний цвет в формате RGB
                    average_color = tuple(map(int, average_color_per_channel))
                    print(average_color)
                    square_size = 200  # Размер квадратика
                    average_color_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)
                    average_color_img[:, :] = average_color  # Заливка квадрата средним цветом

                    # Отображение квадратика
                    cv2.imshow('color',average_color_img)
                    lower_yellow=(150, 150, 0)
                    upper_yellow=(255, 255, 110)
                    
                    lower_red1=(50, 50, 100)
                    upper_red1=(10, 255, 255)
                    lower_red2=(140, 100, 130)
                    upper_red2=(180, 255, 255)
                    # if len(global_dict) >= 2:
                    #     last_value = global_dict[-1][0]       # Последнее значение первого столбца
                    #     second_last_value = global_dict[-2][0] 
                    if all(lower_yellow[i] <= average_color[i] <= upper_yellow[i] for i in range(3)):
                        # print('Лимон')
                        global_dict.append(["лимон",list(average_color)])
                        msg.data=0
                    elif all(lower_red1[i] <= average_color[i] <= upper_red1[i] for i in range(3)) or all(lower_red2[i] <= average_color[i] <= upper_red2[i] for i in range(3)):
                        # print('Перец',np.array(average_color))
                        global_dict.append(["Перец",list(average_color)])

                        msg.data=0
                    else:
                        global_dict.append(["Груша",list(average_color)])
                        # print('Груша',np.array(average_color))
                        msg.data=1
                    sigh_detector.plant_detect.publish(msg)  # Публикуем сообщение
                    rospy.loginfo(f"Published plant_detect: {msg.data}") 
                        
                
            # cv2.imshow("Detected", cv2.resize(camera.frame, (int(camera.width/2), int(camera.height/2)), interpolation = cv2.INTER_NEAREST) ) 
            cv2.imshow("Detected", camera.frame) 
            time.sleep(2)
        # else:
        #     # print("no image")
        if cv2.waitKey(1) == ord('q'):
            break
        
    print(global_dict)
    camera.frame.release()
    cv2.destroyAllWindows()
main()
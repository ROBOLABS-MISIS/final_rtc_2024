#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import rospy
import time
import logging
import os
global_dict=[]
global_dict1=[[2, 0.7044982698961938, 'red'], [9, 0.674034/9658495869, 'red'], [8, 0.706589438546734, 'red'], 
            [5,  0.688798798798789, 'red'], [11,  0.669080932433430433, 'red'], [4,0.704402385290384, 'red']]
# {'1': 16, '2': -5, '3': 11, '9': -17, '4': -13, '8': -8, '10': 17, '5': -10, '11': -10, '12': 17, '15': 7, '7': 18}т
class SighDetector:
    def __init__(self) -> None:
        self.image_sub = None
        self.pub_detect = None
        self.bridge = None
        self.rate = None
        self.image = None
        

    def r_init_node(self):
        rospy.init_node('sigh_detector')
        self.rate = rospy.Rate(0.5)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback)
        self.pub_detect = rospy.Publisher('/sigh_result', String, queue_size=10)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        self.image = cv_image

        # cv.imshow("Image window", self.image)
        # cv.waitKey(1)

    def start(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

        cv2.destroyAllWindows()
        
class Img:
    def __init__(self):
        self.frame=None
        self.grey_img=None
        self.mask=None
        self.crop_with_circle=None
        self.crop_with_aruco=None
        self.height=None
        self.width=None
        
    
 
    
    def detect_aruco(self,check,color):
        if self.crop_with_aruco is not None:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
            parameters = cv2.aruco.DetectorParameters()

            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            cv2.imshow('crop_with_aruco', self.crop_with_aruco)

            if ids is not None:
                # print(ids)
                for id in ids:
                    
                    detected_ids = ids.flatten()
                    for marker_id in detected_ids:
                        if marker_id not in [entry[0] for entry in global_dict]:                    
                            global_dict.append([int(id),check,color])
                    # for id_array in ids:
                    #     id_value = int(id_array[0])  # Преобразуем в int
                    #     if id_value not in global_dict:
                            global_dict.append([int(id),check,color])
                            
                    
                    # if int(id) not in global_dict:
                    #     print(int(id))                  
                    #     global_dict.append([int(id),check,color])
                            # cv2.imshow('crop_with_aruco', self.crop_with_aruco)
                # print(ids)
                        
                cv2.aruco.drawDetectedMarkers(self.crop_with_aruco, corners, ids)
            

        
    def detected_circles(self):
    
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) 
        self.grey_img=cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)[1]


        # img=self.mask
        # Blur using 3 * 3 kernel. 
        # Применяем размытие
        blurred = cv2.blur(self.grey_img, (3, 3)) 

        # Apply Hough transform on the blurred image. 
        #1)изображение в серых градациях, 
        #3)насколько точно будут обнаруживаться круги, 
        #4)Минимальное расстояние между центрами 
        #5)порог используется для определения сильных границ на изображении
        # 6) пороговое значение для центра круга,
        #возвращает координаты
        detected_circles = cv2.HoughCircles(blurred, 
                        cv2.HOUGH_GRADIENT, 1, 100, param1 = 120, 
                    param2 = 40, minRadius = 70, maxRadius = 120) 
#70 100
        # Draw circles that are detected. 
        if detected_circles is not None: 
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles))
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 
                # self.crop_green=self.frame[b-r:b, a-r:a]
                
                self.crop_with_circle=self.frame[max(0,b-r):b, 
                                           max(0,a-r):a]
                self.crop_with_aruco=self.frame[max(0,b-3*r):min(b-r,self.height),
                                                max(0,a-r):min(a+r,self.width)]
            if self.crop_with_circle is not None and self.crop_with_circle.size > 0:
                # cv2.imshow('crop',self.crop_with_circle)
                
                # cv2.imshow('crop',self.crop_with_aruco)
                # cv2.imshow('crop1',self.crop_with_aruco)
                # crop_img = img[y:y+h, x:x+w]
                # Draw the circumference of the circle. 
                cv2.circle(self.frame, (a, b), r, (0, 255, 0), 2) 

                # Draw a small circle (of radius 1) to show the center. 
                cv2.circle(self.frame, (a, b), 1, (0, 0, 255), 3) 
            
    def green_mask(self):
        # Преобразование изображения в цветовое пространство HSV
        if self.crop_with_circle is not None:
            hsv_img = cv2.cvtColor(self.crop_with_circle, cv2.COLOR_BGR2HSV)

            # lower_green = np.array([33, 69, 32], dtype = "uint8")
            # upper_green = np.array([162, 221, 240], dtype = "uint8")
            
            lower_green = np.array([68, 54, 62], dtype = "uint8")
            
            upper_green = np.array([162, 244, 242], dtype = "uint8")
                
            #применяем маску
            self.mask_G= cv2.inRange(hsv_img, lower_green, upper_green)
            
            # Применение морфологических операций для удаления шумов
            kernel = np.ones((5, 5), np.uint8)
            self.mask_G = cv2.morphologyEx(self.mask_G, cv2.MORPH_CLOSE, kernel)  # Закрытие (удаление маленьких черных дырок)
            self.mask_G= cv2.morphologyEx(self.mask_G, cv2.MORPH_OPEN, kernel)   # Открытие (удаление шумов)
            # print(self.mask_G)
            # print(cv2.countZero(self.mask_G))
            height, width = self.mask_G.shape[:2] 
            check=cv2.countNonZero(self.mask_G)/(height* width)
            # print(check)

            # print()

            if check>0.3:
                if check>0.6 :
                    color="red"
                            
                else:
                    color="green"
                self.detect_aruco(check,color) 
                
            # cv2.imshow("Green Mask", self.mask_G)  # Показываем саму маску
            
    def detect_qr(self):
        log_file_path="/home/nastya314/catkin_ws/src/final_rtc/qrcode_log.txt"
        # Если лог-файл существует, он будет очищен перед запуском
        # if os.path.exists(log_file_path):
        #     os.remove(log_file_path)
        # Настройка логгирования
        logging.basicConfig(
            # filename="qrcode_log.txt",
            
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s - QR-код: %(message)s",
        )

        # Инициализация камеры
        detector = cv2.QRCodeDetector()

        # Обнаруживаем и декодируем QR-код
        data, bbox, _ = detector.detectAndDecode(self.frame)

        # Если QR-код найден
        if bbox is not None:
            if data:  # Если данные считаны
                # print(f"Обнаружен QR-код: {data}")
                logging.info(data)  # Запись в лог-файл

            # Рисуем рамку вокруг QR-кода
            for i in range(len(bbox)):
                point1 = tuple(map(int, bbox[i][0]))
                point2 = tuple(map(int, bbox[(i + 1) % len(bbox)][0]))
                cv2.line(self.frame, point1, point2, color=(0, 255, 0), thickness=2)
    
def main():
    # print(global_dict1)
    sigh_detector = SighDetector()
    sigh_detector.r_init_node()
    print("start")
    
    # time.sleep(5)
    camera=Img()
    global global_dict
    global_dict=[]
    # cap=cv2.VideoCapture("/home/nastya314/Downloads/slow1.mp4")
    # cap=cv2.VideoCapture(2)
    
    # frame_width = int(cap.get(3)) 
    # frame_height = int(cap.get(4)) 
   
    # size = (frame_width, frame_height) 
    # result = cv2.VideoWriter('/home/nastya314/Downloads/video.mp4',  
    #                                 cv2.VideoWriter_fourcc(*'MP4V'), 
    #                                 20, size) 
    if __name__ == '__main__':
        def nothing(*arg):
            pass

    
    while not rospy.is_shutdown():
        if sigh_detector.image is not None:
    # while True:
    #     if cap.isOpened():
        # if f==1:
            camera.frame = sigh_detector.image
            # camera.frame=cv2.rotate(sigh_detector.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # ret, camera.frame=cap.read()
            # camera.frame=cv2.imread("/home/nastya314/Downloads/new.png")
            camera.height, camera.width = camera.frame.shape[:2] 
            
            # result.write(camera.frame) 
            # Below VideoWriter object will create 
            # a frame of above defined The output  
            # is stored in 'filename.avi' file. 
            
            
            # camera.setting_mask()
            
            camera.detected_circles()
            # camera.detect_aruco()
            if camera.crop_with_circle is not None and camera.crop_with_circle.size > 0:
                camera.green_mask()      
            camera.detect_qr()
            # cv2.imshow("Detected", cv2.resize(camera.frame, (int(camera.width/2), int(camera.height/2)), interpolation = cv2.INTER_NEAREST) ) 
            # cv2.imshow("Thresh", cv2.resize(camera.grey_img, (int(camera.width/2), int(camera.height/2)), interpolation = cv2.INTER_NEAREST) ) 
            cv2.imshow("Detected Circle", camera.frame) 
            # cv2.imshow("Thresh", camera.grey_img) 
            # cv2.imshow("Grey image", camera.grey_img) 
            # cv2.imshow("Mask image", camera.mask) 
        # else:
        #     print("no image")
            time.sleep(1)
        
        if cv2.waitKey(1) == ord('q'):
            print(global_dict1)
            
            break
        
    
    camera.frame.release()
    cv2.destroyAllWindows()
main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import logging
import os
log_file_path="/home/nastya314/catkin_ws/src/final_rtc/qrcode_log.txt"
# Если лог-файл существует, он будет очищен перед запуском
if os.path.exists(log_file_path):
    os.remove(log_file_path)
# Настройка логгирования
logging.basicConfig(
    # filename="qrcode_log.txt",
    
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - QR-код: %(message)s",
)

# Инициализация камеры
cap = cv2.VideoCapture(0)
detector = cv2.QRCodeDetector()

print("Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка доступа к камере.")
        break

    # Обнаруживаем и декодируем QR-код
    data, bbox, _ = detector.detectAndDecode(frame)

    # Если QR-код найден
    if bbox is not None:
        if data:  # Если данные считаны
            print(f"Обнаружен QR-код: {data}")
            logging.info(data)  # Запись в лог-файл

        # Рисуем рамку вокруг QR-кода
        for i in range(len(bbox)):
            point1 = tuple(map(int, bbox[i][0]))
            point2 = tuple(map(int, bbox[(i + 1) % len(bbox)][0]))
            cv2.line(frame, point1, point2, color=(0, 255, 0), thickness=2)

    # Отображаем кадр
    cv2.imshow("QR Code Scanner", frame)

    # Проверяем нажатие клавиши 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

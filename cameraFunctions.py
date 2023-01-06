import numpy as np
from cameraParams import *
from time import time
import arucoConfig as ac


def find_markers(frame, showImg, show=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, DICTIONARY, parameters=PARAMETERS)
    if show:
        cv2.aruco.drawDetectedMarkers(showImg, corners, ids)

    return corners, ids, showImg


def find_ball(frame, corners, show=False):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ballTime = time()
    blankImage = np.zeros((ac.height, ac.width, 1), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel_close)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel_open)

    blankImage[dst > 0.01 * dst.max()] = 255

    blankImage = cv2.medianBlur(blankImage, 5, 0)

    rows = blankImage.shape[0]

    # Закрашиваем aruco метки, чтобы они не попадали под кандидаты в мячи
    if len(corners) > 0:
        for i in range(len(corners)):
            pts = np.array([[
                [corners[i][0][0][0], corners[i][0][0][1]],
                [corners[i][0][1][0], corners[i][0][1][1]],
                [corners[i][0][2][0], corners[i][0][2][1]],
                [corners[i][0][3][0], corners[i][0][3][1]],
            ]], np.int32)
            cv2.fillPoly(blankImage, [pts], (0, 0, 0))

    # Ищем круги по контурам при помощи HoughCircles. Отбор кандидатов.
    circles = None
    circles = cv2.HoughCircles(blankImage.copy(), cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=1, minRadius=1,
                               maxRadius=4)
    result = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Для каждого кандидата считаем количество "блобов". В зависимости от их кол-ва и наличия, отбрасываем или приниаем кандидата. 
        for circle in circles[0, :]:
            try:
                delta = round(float(circle[2]) * 2)
                ballImage = frame[int(circle[1]) - delta:int(circle[1]) + delta,
                            int(circle[0]) - delta:int(circle[0]) + delta]
                ballImage = cv2.resize(ballImage, (48, 48))
                ballKeypoints = ballDetector.detect(ballImage)

                if len(ballKeypoints) > 0:
                    if show:
                        cv2.circle(frame, (int(circle[0]), int(circle[1])), delta, (255, 0, 0), 2)

                    result.append((int(circle[0]), int(circle[1])))
            except Exception as e:
                pass

    return result, frame


def calcArucoCenter(points):
    _x_list = [point[0] for point in points]
    _y_list = [point[1] for point in points]
    _len = len(points)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len

    return _x, _y

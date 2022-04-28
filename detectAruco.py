import cv2
import logging
import math
import time
import numpy as np
import paho.mqtt.client as paho
import json
from collections import deque

import os

import arucoConfig as ac

cmd = "v4l2-ctl --set-ctrl=auto_exposure={0} --set-ctrl=exposure_time_absolute={1} --set-ctrl=brightness={2} " \
      "--set-ctrl=iso_sensitivity=1 "
cmd = cmd.format(ac.autoExposure, ac.exposureTime, ac.brightness)
print(cmd)

with open("baseCoordinates.json", "r", encoding="utf-8") as file:
    baseCoordinates = json.load(file)

client = paho.Client()
client.username_pw_set(ac.mqtt_login, ac.mqtt_pwd)
client.connect(host=ac.hostName)

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

PARAMETERS = cv2.aruco.DetectorParameters_create()
MARKER_EDGE = 0.05

buffer = 64

pts = deque(maxlen=buffer)

# Canny Edge Detection
kernel = np.array((
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0]), dtype='uint8')
# kernel=kernel/4
kernelEr = np.ones((3, 3), 'uint8')
kernelEr[0][0] = 0
kernelEr[0][2] = 0
kernelEr[2][0] = 0
kernelEr[2][2] = 0

image_number = 0

delta = 4
delta1 = 7
delta_mouse = 20

crossSize = 10

changeDelta = 30

dist = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
dist_less = lambda x1, y1, x2, y2, d: dist(x1, y1, x2, y2) <= d
dist_line_less = lambda x1, y1, x2, y2, x0, y0, d: abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) <= d * dist(
    x1, y1, x2, y2)


def sendMarkers(topic, msg):
    # publish.single(topic, json.dumps(msg), hostname=hostName, auth={'username' : mqtt_login, 'password': mqtt_pwd})
    client.publish(topic, json.dumps(msg))


def angles_from_rvec(rvec):
    r_mat, _jacobian = cv2.Rodrigues(rvec)
    a = math.atan2(r_mat[2][1], r_mat[2][2])
    b = math.atan2(-r_mat[2][0], math.sqrt(math.pow(r_mat[2][1], 2) + math.pow(r_mat[2][2], 2)))
    c = math.atan2(r_mat[1][0], r_mat[0][0])
    return [a, b, c]


def calc_heading(rvec):
    angles = angles_from_rvec(rvec)
    degree_angle = math.degrees(angles[2])
    if degree_angle < 0:
        degree_angle = 360 + degree_angle
    return degree_angle


def find_markers(frame, show=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, DICTIONARY, parameters=PARAMETERS)
    if show:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_EDGE, ac.CAMERA_MATRIX,
    # ac.DIST_COEFFS)
    return corners, ids


kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def find_ball(frame, corners, imgBig, show=False):
    global kernel_close, kernel_open, previousCircle, image_number
    blank_image = np.zeros((ac.height, ac.width, 1), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dst = cv2.morphologyEx(cv2.Canny(gray, 100, 100), cv2.MORPH_CLOSE, kernel_close)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel_open)
    blank_image[dst > 0.05 * dst.max()] = 255
    # mask = cv2.medianBlur(blank_image, 3)
    mask = blank_image
    rows = mask.shape[0]
    if len(corners) > 0:
        for i in range(len(corners)):
            pts = np.array([[
                [corners[i][0][0][0], corners[i][0][0][1]],
                [corners[i][0][1][0], corners[i][0][1][1]],
                [corners[i][0][2][0], corners[i][0][2][1]],
                [corners[i][0][3][0], corners[i][0][3][1]],
            ]], np.int32)
            cv2.fillPoly(mask, [pts], (0, 0, 0))
    circles = None
    circles = cv2.HoughCircles(mask.copy(), cv2.HOUGH_GRADIENT, 1, rows / 8, param1=50, param2=7, minRadius=2,
                               maxRadius=15)
    cv2.imshow("mask", mask)
    scaleRatio = 1280 // ac.width
    if circles is not None:
        circles = np.uint16(np.around(circles))
        i_ = 0
        for circle in circles[0, :]:
            if show:
                cv2.circle(frame, (circle[0], circle[1]), 1, (0, 100, 100), 3)
                cv2.circle(frame, (circle[0], circle[1]), circle[2], (255, 0, 255), 3)
        # return circle
        return circles[0, :]
    # print(circles)

    return ()


def check_point(x, y):
    found = False
    for crossPoint in crossPoints:
        if dist_less(x, y, crossPoint[0], crossPoint[1], delta1):
            crossPoint[0] = (crossPoint[0] * crossPoint[2] + x) / (crossPoint[2] + 1)
            crossPoint[1] = (crossPoint[1] * crossPoint[2] + y) / (crossPoint[2] + 1)
            crossPoint[2] = crossPoint[2] + 1
            found = True

    if not found:
        crossPoints.append([x, y, 1, 0])


def draw(img):
    cdstP = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for crossPoint in crossPoints:
        if crossPoint[3] > 0:
            colour = (0, 255, 0)
        else:
            colour = (255, 255, 255)
        cv2.line(cdstP, (round(crossPoint[0] * 2) + crossSize, round(crossPoint[1] * 2) + crossSize),
                 (round(crossPoint[0] * 2) - crossSize, round(crossPoint[1] * 2) - crossSize), colour, 2,
                 cv2.LINE_AA)
        cv2.line(cdstP, (round(crossPoint[0] * 2) + crossSize, round(crossPoint[1] * 2) - crossSize),
                 (round(crossPoint[0] * 2) - crossSize, round(crossPoint[1] * 2) + crossSize), colour, 2,
                 cv2.LINE_AA)
        if crossPoint[3] > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cdstP, str(crossPoint[3]), (round(crossPoint[0] * 2) + 3, round(crossPoint[1] * 2)),
                        font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("lines", cdstP)


def findLines(img):
    global kernel
    global kernelEr
    global crossPoints
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)  # Canny Edge Detection
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernelEr, iterations=6)
    cdstP = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, None, 100, 30)

    crossPoints = list()
    if linesP is not None:
        print(len(linesP))

        for i in range(0, len(linesP)):
            line = linesP[i][0]
            if abs(line[3] - line[1]) > abs(line[2] - line[0]):
                for j in range(0, len(linesP)):
                    l1 = linesP[j][0]

                    if abs(l1[3] - l1[1]) <= abs(l1[2] - l1[0]):
                        x11 = line[0]
                        x12 = line[2]
                        y11 = line[1]
                        y12 = line[3]
                        x21 = l1[0]
                        x22 = l1[2]
                        y21 = l1[1]
                        y22 = l1[3]
                        if dist_less(x11, y11, x21, y21, delta):
                            check_point(round((x11 + x21) / 2), round((y11 + y21) / 2))
                        if dist_less(x11, y11, x22, y22, delta):
                            check_point(round((x11 + x22) / 2), round((y11 + y22) / 2))
                        if dist_less(x12, y12, x21, y21, delta):
                            check_point(round((x12 + x21) / 2), round((y12 + y21) / 2))
                        if dist_less(x12, y12, x22, y22, delta):
                            check_point(round((x12 + x22) / 2), round((y12 + y22) / 2))
                        if dist_line_less(x11, y11, x12, y12, x21, y21, delta):
                            check_point(x21, y21)
                        if dist_line_less(x11, y11, x12, y12, x22, y22, delta):
                            check_point(x22, y22)
                        if dist_line_less(x21, y21, x22, y22, x11, y11, delta):
                            check_point(x11, y11)
                        if dist_line_less(x21, y21, x22, y22, x12, y12, delta):
                            check_point(x12, y12)

    # draw(img)

    # lines = []
    lines = {'camId': ac.camId,
             'lines': []}
    crossPoints_ = crossPoints.copy()
    for baseCoordinate in baseCoordinates['points']:
        okPoints = []
        okPointsDelta = []
        for crossPoint in crossPoints_:
            d1_ = abs(float(baseCoordinate[0]) - float(crossPoint[0]))
            d2_ = abs(float(baseCoordinate[1]) - float(crossPoint[1]))
            if d1_ <= changeDelta and \
                    d2_ <= changeDelta:
                okPoints.append(crossPoint[:2])
                okPointsDelta.append((d1_ + d2_) / 2)
                crossPoints_.remove(crossPoint)
        if len(okPointsDelta) > 0:
            lines['lines'].append(okPoints[okPointsDelta.index(min(okPointsDelta))])
    try:
        client.publish("MIPT-SportRoboticsClub/LunokhodFootball/PitchLines", json.dumps(lines))
    except TypeError:
        print(lines)


def capture():
    os.system(cmd)

    if ac.showFrame:
        print("Show")
        cv2.namedWindow("input")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, ac.framerate)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Frame Rate: ", frame_rate)
    print("Height: ", height)
    print("Width: ", width)
    time.sleep(2.0)

    logging.info("Start capturing from pi camera.")
    try:
        frame_num = 0
        time1 = 0
        time2 = 0
        while True:
            frame_num += 1
            print("FPS:", 1 / (time.time() - time1))
            time1 = time.time()
            ret, imgBig = cap.read()
            img = cv2.resize(imgBig, (ac.width, ac.height), interpolation=cv2.INTER_AREA)
            corners, ids = find_markers(img.copy(), ac.showFrame)
            jsonMarkers = """{
			"markers":[""" + "{}," * (len(corners) - 1) + "{}" + """],
			"camId" : """ + str(ac.camId) + "}"

            markers = json.loads(jsonMarkers)
            markers['count'] = len(corners)

            if len(corners) > 0:
                for i in range(0, len(corners)):  # если найден хоть один маркер
                    markers['markers'][i] = {'marker-id': int(ids[i][0]), 'camId': ac.camId,
                                             'corners': {'1': {'x': float(corners[i][0][0][0]),
                                                               'y': float(corners[i][0][0][1])},
                                                         '2': {'x': float(corners[i][0][1][0]),
                                                               'y': float(corners[i][0][1][1])},
                                                         '3': {'x': float(corners[i][0][2][0]),
                                                               'y': float(corners[i][0][2][1])},
                                                         '4': {'x': float(corners[i][0][3][0]),
                                                               'y': float(corners[i][0][3][1])}
                                                         }}
            sendMarkers(ac.topicRoot + ac.camId, markers)
            # balls = find_ball(img, corners, imgBig, ac.showFrame)
            # if len(balls) > 0:
            #     ball = {'camId': ac.camId, 'ball': []}
            #     for b in balls:
            #         ball['ball'].append({'center': {'x': float(b[0]), 'y': float(b[1])}})
            # else:
            #     ball = {'camId': ac.camId, 'ball': 'None'}
            # sendMarkers(ac.topicBall + ac.camId, ball)
            if ac.showFrame:
                cv2.imshow("input", img)
            if frame_num % 12 == 0:
                findLines(img)
            key = cv2.waitKey(1)
            if key == 27:
                break

    except Exception as e:
        logging.error("Capturing stopped with an error:" + str(e))
    finally:
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()


if __name__ == "__main__":
    capture()

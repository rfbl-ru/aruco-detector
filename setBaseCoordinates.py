# -*- coding: utf-8 -*-

import cv2
import sys
import math
import numpy as np
import json
import os
import arucoConfig as ac

# img = cv2.imread("pic2.jpg")

cmd = "v4l2-ctl --set-ctrl=auto_exposure={0} --set-ctrl=exposure_time_absolute={1} --set-ctrl=brightness={2} --set-ctrl=iso_sensitivity=1"
cmd = cmd.format(ac.autoExposure, ac.exposureTime, ac.brightness)
print(cmd)

os.system(cmd)

cap = cv2.VideoCapture(0)
ret, img = cap.read()


if img is None:
    sys.exit("Could not read the image.")
# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
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
edges = cv2.Canny(image=img_gray, threshold1=10, threshold2=200)  # Canny Edge Detection
edges = cv2.dilate(edges, kernel, iterations=2)
edges = cv2.erode(edges, kernelEr, iterations=6)
cdstP = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, None, 100, 30)

# размер области близости точек
# концы отрезков горизонтальных и вертикальных линий считаются близкими если ближе delta
delta = 4
# несколько найденных точек считаются одной, если они ближе delta1, берётся их центр масс
delta1 = 7
# при работе курсором считаем курсор в окрестности точки, если ближе delta_mouse
delta_mouse = 20

# размер крестика на картинке, обозначающего точку
crossSize = 10

# список точек с учётом их близости
dist = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
dist_less = lambda x1, y1, x2, y2, d: dist(x1, y1, x2, y2) <= d
dist_line_less = lambda x1, y1, x2, y2, x0, y0, d: abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) <= d * dist(
    x1, y1, x2, y2)

crossPoints = list()


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
                    # проверяем расстояния между концами отрезков
                    if dist_less(x11, y11, x21, y21, delta):
                        check_point(round((x11 + x21) / 2), round((y11 + y21) / 2))
                    if dist_less(x11, y11, x22, y22, delta):
                        check_point(round((x11 + x22) / 2), round((y11 + y22) / 2))
                    if dist_less(x12, y12, x21, y21, delta):
                        check_point(round((x12 + x21) / 2), round((y12 + y21) / 2))
                    if dist_less(x12, y12, x22, y22, delta):
                        check_point(round((x12 + x22) / 2), round((y12 + y22) / 2))
                    # 1-й конец l1 с отрезком l
                    if dist_line_less(x11, y11, x12, y12, x21, y21, delta):
                        check_point(x21, y21)
                    # 2-й конец l1 с отрезком l
                    if dist_line_less(x11, y11, x12, y12, x22, y22, delta):
                        check_point(x22, y22)
                    # 1-й конец l с отрезком l1
                    if dist_line_less(x21, y21, x22, y22, x11, y11, delta):
                        check_point(x11, y11)
                    # 2-й конец l с отрезком l1
                    if dist_line_less(x21, y21, x22, y22, x12, y12, delta):
                        check_point(x12, y12)

crossPoints.sort()
print("crossPoints", crossPoints)


def draw():
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


point_index = 0
redraw = True


def click_mark_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for cp in crossPoints:
            if dist_less(cp[0], cp[1], x / 2, y / 2, delta_mouse) and (cp[3] == 0):
                global point_index
                point_index += 1
                cp[3] = point_index
                print(cp)
                draw()


cv2.namedWindow("lines")
cv2.setMouseCallback("lines", click_mark_point)

while True:
    if redraw:
        draw()
        redraw = False
    key = cv2.waitKey(0)
    if key > 0:
        break

cv2.destroyAllWindows()

imgPoints = np.zeros((8, 2))

for cp in crossPoints:
    imgPoints[cp[3] - 1][0] = cp[0]
    imgPoints[cp[3] - 1][1] = cp[1]

jsonData = {
    "points": []
}
# print()
for point in imgPoints:
    coordinates = []
    for coordinate in point:
        coordinates.append(float(coordinate))
    if coordinates[0] != 0 and coordinates[1] != 0:
        jsonData['points'].append(coordinates)

with open("baseCoordinates.json", "w+", encoding='utf-8') as file:
    json.dump(jsonData, file)

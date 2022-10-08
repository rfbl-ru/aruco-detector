import cv2
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


# def find_ball(frame, show=False):
#     ballTime = time()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.medianBlur(frame, 5)
#     ret, thresh_img = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)

#     # cv2.imshow("Bin", thresh_img)

#     dst = cv2.Canny(thresh_img.copy(), 100, 100)

#     # dst = cv2.blur(dst, (3, 3))

#     keypoints = detector.detect(dst)

#     imageWithKeypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (255, 0, 0), 
#         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
#     cv2.imshow("cand", dst)

#     result = []

#     if len(keypoints) > 0:
#         for keypoint in keypoints:
#             try:
#                 delta = round(keypoint.size)
#                 ballImage = frame[int(keypoint.pt[1]) - delta:int(keypoint.pt[1]) + delta,
#                         int(keypoint.pt[0]) - delta:int(keypoint.pt[0]) + delta]
#                 ballImage = cv2.resize(ballImage, (48, 48))
#                 ballKeypoints = ballDetector.detect(ballImage)

#                 if len(ballKeypoints) > 0:
#                     if show:
#                         cv2.circle(frame, (int(keypoint.pt[0]), int(keypoint.pt[1])), round(keypoint.size), (255, 0, 0), 2)

#                     result.append((int(keypoint.pt[0]), int(keypoint.pt[1])))
#                 imageWithKeypoints = cv2.drawKeypoints(dst, keypoints, np.array([]), (255, 0, 0), 
#                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#                 cv2.imshow("blobs", imageWithKeypoints)
#                 ballImageWithKeypoints = cv2.drawKeypoints(ballImage, ballKeypoints, np.array([]), (255, 0, 0),
#                                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#                 cv2.imshow("ballCand", ballImageWithKeypoints)

                
#             except Exception as e:
#                 pass
#     print("candTime:{}".format(time() - ballTime))
#     # ballTime = time()
#     return result, frame


def find_ball(frame, showImg, corners, show=False):
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



    if len(corners) > 0:
        for i in range(len(corners)):
            pts = np.array([[
                [corners[i][0][0][0], corners[i][0][0][1]],
                [corners[i][0][1][0], corners[i][0][1][1]],
                [corners[i][0][2][0], corners[i][0][2][1]],
                [corners[i][0][3][0], corners[i][0][3][1]],
            ]], np.int32)
            cv2.fillPoly(blankImage, [pts], (0, 0, 0))
    # circlesTime = time()
    # cv2.imshow("bi",blankImage)
    circles = None
    circles = cv2.HoughCircles(blankImage.copy(), cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=1, minRadius=1, maxRadius=4)
    # print("CriclesTime: {}".format(time() - circlesTime))
    result = []

    # cv2.imshow("circles", blankImage)
    # print(circles is None)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            try:
                delta = round(float(circle[2])*2)
                ballImage = frame[int(circle[1]) - delta:int(circle[1]) + delta,
                        int(circle[0]) - delta:int(circle[0]) + delta]
                ballImage = cv2.resize(ballImage, (48, 48))
                ballKeypoints = ballDetector.detect(ballImage)

                if len(ballKeypoints) > 0:
                    if show:
                        cv2.circle(showImg, (int(circle[0]), int(circle[1])), delta, (255, 0, 0), 2)

                    result.append((int(circle[0]), int(circle[1])))
                # imageWithKeypoints = cv2.drawKeypoints(dst, keypoints, np.array([]), (255, 0, 0), 
                #         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # cv2.imshow("blobs", imageWithKeypoints)
                # ballImageWithKeypoints = cv2.drawKeypoints(ballImage, ballKeypoints, np.array([]), (255, 0, 0),
                                                   # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # cv2.imshow("ballCand", cv2.resize(ballImage, (100, 100)))
                
                
            except Exception as e:
                # print(e)
                pass
    # print("candTime:{}".format(time() - ballTime))
    # ballTime = time()
    return result, frame, showImg


def calcArucoCenter(points):
    _x_list = [point[0] for point in points]
    _y_list = [point[1] for point in points]
    _len = len(points)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len

    return (_x, _y)


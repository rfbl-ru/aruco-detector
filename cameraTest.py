from __future__ import print_function
import sys
import cv2
# import time
import os
import numpy as np
from cameraFunctions import *
from mqttFunctions import *
import arucoConfig as ac
from pitchCalibration import *

D = 30
L = 293


def main(argv):
    #capture from camera at location 0
    cmd = "v4l2-ctl --set-ctrl=auto_exposure={0} --set-ctrl=exposure_time_absolute={1} --set-ctrl=brightness={2} " \
      "--set-ctrl=iso_sensitivity=1 "
    cmd = cmd.format(ac.autoExposure, ac.exposureTime, ac.brightness)
    os.system(cmd)
    cv2.namedWindow("input")
    cap = cv2.VideoCapture(0)
    
    cap.set( cv2.CAP_PROP_FRAME_HEIGHT, ac.height )
    cap.set( cv2.CAP_PROP_FRAME_WIDTH, ac.width )
    cap.set(cv2.CAP_PROP_FPS, ac.framerate)
    # Read the current setting from the camera
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Frame Rate: ", frame_rate)
    print("Height: ", height)
    print("Width: ", width)
    M = None
    for i in range(100):
        ret, img = cap.read()
        isResult, M = calcPerspectiveMatrix(img)
        if isResult:
            print(M)
            break
    # if len(M) <= 0:
    #     print("Can`t find corner markers!")
    #     return 0
    
    img_number = 0

    while True:
        time1 = time()
        ret, img = cap.read()
        showImg = img.copy()
        arucoImg = img.copy()
        # print("capTime:{}".format(time() - time1))
        # time1 = time()
        
        corners, ids, arucoImg = find_markers(img, arucoImg, ac.showFrame)
        # print("markersTime:{}".format(time() - time1))
        # time1 = time()

        balls, showImg = find_ball(img, corners, ac.showFrame)
        # print("ballTime:{}".format(time() - time1))
        # time1 = time()

        ballAll = []
        cornersAll = []
        for corner in corners:
            cornerResult = []
            for i in range(4):
                cornerPts = np.float32([corner[0][i]])
                cornerPts = np.array([cornerPts])
            # print(corner[0][0])
            # print(corner[0][0][0], corner[0][0][1])
            # corner[0][0][0] = cv2.perspectiveTransform(corner[0][0][0], M)
                cornerResult.append(cv2.perspectiveTransform(cornerPts, M))
            # print(corner, cornerResult)
            cornersAll.append(cornerResult)

        for ball in balls:
            ballPts = np.float32([ball])
            ballPts = np.array([ballPts])

            ballResult = cv2.perspectiveTransform(ballPts, M)
            # ballResult[1] = 283 - ballResult[1]
            y = ballResult[0][0][1]
            # print(y)
            # if y < (L/2 - D/2):
                # w = 1
            # elif y > (L/2 + D/2):
                # w = 0
            # else:
                # if ac.camId == "1":
                    # w = 1 - ((y - (L/2 - D/2))/D)
                # else:
                    # w = (y - (L/2 - D/2))/D
            # ballResult[0][0][1] = ballResult[0][0][1]*w
            # ballResult[0][0][1] = 
            # print(ballResult)
            ballAll.append(ballResult[0][0])
        # print(calcArucoCenter(cv2.perspectiveTransform(ball, M)))
        # print(list(ballAll))
        # print(cornersAll) 
        # print(cornersAll[0][0][0][0][0])
        # print(cornersAll[0][0][0][0][1])
        # print(cornersAll[0][1][0][0][0])
        # print(cornersAll[0][1][0][0][1])
        sendMQTTData(cornersAll, ids)
        if len(ballAll) <= 1:
            sendBallData(ballAll)
        # print("Total time: {}".format(time() - time1))
        cv2.imshow("input", showImg)
        key = cv2.waitKey(1)
        if key == 27:
            break 

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main(sys.argv) 
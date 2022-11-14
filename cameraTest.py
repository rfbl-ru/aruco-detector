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
from time import sleep

pitchDelta = 5


def main(argv):
    #capture from camera at location 0
    cmd = "v4l2-ctl --device /dev/video{0} --set-ctrl=exposure_auto={1} --set-ctrl=exposure_absolute={2} --set-ctrl=brightness=200 "
    cmd = cmd.format(ac.linuxCameraNum, ac.autoExposure, ac.exposureTime)
    os.system(cmd)
    sleep(1)
    os.system("v4l2-ctl --device /dev/video{0} --set-ctrl=brightness={1}".format(ac.linuxCameraNum, ac.brightness))
    cv2.namedWindow("input")
    cap = cv2.VideoCapture(ac.linuxCameraNum)
    
    cap.set( cv2.CAP_PROP_FRAME_HEIGHT, ac.height )
    cap.set( cv2.CAP_PROP_FRAME_WIDTH, ac.width )
    cap.set(cv2.CAP_PROP_FPS, ac.framerate)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Frame Rate: ", frame_rate)
    print("Height: ", height)
    print("Width: ", width)
    M = None
    isResult = False
    for i in range(1000):
        ret, img = cap.read()
        isResult, M, pitchCorners = calcPerspectiveMatrix(img)
        if isResult:
            print(M)
            break
    if not isResult:
        print("Can`t find corner markers!")
        return 0
    
    img_number = 0


    # TODO: Протестировать! 
    cornerMax = 0
    cornerMin = 10000
    pitchIds = [0, 1, 2, 3]


    for i in range(4):
        _sum = pitchCorners[i][0] + pitchCorners[i][1]
        if (_sum) > cornerMax:
            cornerMaxId = i
            cornerMax = _sum
        if (_sum) < cornerMin:
            cornerMinId = i
            cornerMin = _sum

    pitchCorners[cornerMaxId][0] += pitchDelta
    pitchCorners[cornerMaxId][1] += pitchDelta
    pitchIds.remove(cornerMaxId)

    pitchCorners[cornerMinId][0] -= pitchDelta
    pitchCorners[cornerMinId][1] -= pitchDelta
    pitchIds.remove(cornerMinId)

    for pId in pitchIds:
        if pitchCorners[pId][0] > pitchCorners[pId][1]:
            pitchCorners[pId][0] += pitchDelta
            pitchCorners[pId][1] -= pitchDelta
        else:
            pitchCorners[pId][0] -= pitchDelta
            pitchCorners[pId][1] += pitchDelta

    pitchCorners.sort()
    buffer = pitchCorners[3]
    pitchCorners[3] = pitchCorners[2]
    pitchCorners[2] = buffer

    while True:
        time1 = time()
        ret, img = cap.read()

        # Закрашивание области за границами игрового поля
        stencil = np.zeros(img.shape).astype(img.dtype)
        pts = np.array([pitchCorners], np.int32)
        cv2.fillPoly(stencil, [pts], (255, 255, 255))

        img = cv2.bitwise_and(img, stencil)

        showImg = img.copy()
        arucoImg = img.copy()


        
        corners, ids, arucoImg = find_markers(img, arucoImg, ac.showFrame)

        balls, showImg = find_ball(img, corners, ac.showFrame)

        ballAll = []
        cornersAll = []
        for corner in corners:
            cornerResult = []
            for i in range(4):
                cornerPts = np.float32([corner[0][i]])
                cornerPts = np.array([cornerPts])
                cornerResult.append(cv2.perspectiveTransform(cornerPts, M))
            cornersAll.append(cornerResult)

        for ball in balls:
            ballPts = np.float32([ball])
            ballPts = np.array([ballPts])

            ballResult = cv2.perspectiveTransform(ballPts, M)
            y = ballResult[0][0][1]
            ballAll.append(ballResult[0][0])
        sendMQTTData(cornersAll, ids)
        if len(ballAll) <= 1:
            sendBallData(ballAll)
        cv2.imshow("input", showImg)
        key = cv2.waitKey(1)
        if key == 27:
            break 

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main(sys.argv) 

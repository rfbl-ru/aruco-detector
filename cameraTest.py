from __future__ import print_function
import sys
import cv2
import os
import numpy as np
from cameraFunctions import *
from mqttFunctions import *
import arucoConfig as ac
from pitchCalibration import *

D = 30
L = 293


def main(argv):
    # Устанавливаем настройки камеры
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
    
    # Калибровка по угловым меткам поля для пересчёта в мировые координаты
    for i in range(100):
        ret, img = cap.read()
        isResult, M = calcPerspectiveMatrix(img)
        if isResult:
            print(M)
            break
    
    img_number = 0

    while True:
        time1 = time()
        ret, img = cap.read()
        showImg = img.copy()
        arucoImg = img.copy()
        
        corners, ids, arucoImg = find_markers(img, arucoImg, ac.showFrame) 

        balls, showImg = find_ball(img, corners, ac.showFrame) 

        ballAll = []
        cornersAll = []
        # Пересчёт координат для aruco меток
        for corner in corners:
            cornerResult = []
            for i in range(4):
                cornerPts = np.float32([corner[0][i]])
                cornerPts = np.array([cornerPts])
                cornerResult.append(cv2.perspectiveTransform(cornerPts, M))
            cornersAll.append(cornerResult)
        
        # Пересчёт координат для мяча
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

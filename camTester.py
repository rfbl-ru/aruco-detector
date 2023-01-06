from pitchCalibration import *
from math import ceil
import os
import sys


def main(argv):
    cmd = "v4l2-ctl --device /dev/video{0} --set-ctrl=exposure_auto={1} --set-ctrl=exposure_absolute={2} " \
          "--set-ctrl=brightness={3}"
    cmd = cmd.format(ac.linuxCameraNum, ac.autoExposure, ac.exposureTime, ac.brightness)
    os.system(cmd)
    cv2.namedWindow("input")
    cap = cv2.VideoCapture(ac.linuxCameraNum)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ac.height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ac.width)
    cap.set(cv2.CAP_PROP_FPS, ac.framerate)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Frame Rate: ", frame_rate)
    print("Height: ", height)
    print("Width: ", width)

    framesTotalSum, framesTotalCount = 0, 0
    framesSecondSum, framesSecondCount = 0, 0
    timeStart = time()
    prevTime = timeStart
    while True:

        ret, img = cap.read()
        arucoImg = img.copy()

        corners, ids, arucoImg = find_markers(img, arucoImg, ac.showFrame)
        balls, showImg = find_ball(img, corners, ac.showFrame)

        framesSecondSum += 1
        timeSinceLast = time() - prevTime
        if timeSinceLast >= 1:
            framesSecondCount = int(ceil(framesSecondSum / timeSinceLast))
            print("FPS:{}".format(framesSecondCount))
            framesTotalSum += framesSecondCount
            framesSecondSum = 0
            prevTime = time()

        timeSinceStart = time() - timeStart
        if timeSinceStart >= 30:
            framesTotalCount = int(ceil(framesTotalSum / timeSinceStart))
            print("\nTotal FPS:{}".format(framesTotalCount))
            break

        cv2.imshow("input", showImg)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main(sys.argv)

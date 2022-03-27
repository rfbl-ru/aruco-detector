import asyncio
import cv2
import concurrent
import logging
import math
import time
from imutils.video import FPS
import numpy as np
import imutils
import paho.mqtt.client as paho
import json
from collections import deque

import os

import arucoConfig as ac


cmd = "v4l2-ctl --set-ctrl=auto_exposure={0} --set-ctrl=exposure_time_absolute={1} --set-ctrl=brightness={2} --set-ctrl=iso_sensitivity=1"
cmd = cmd.format(ac.autoExposure, ac.exposureTime, ac.brightness)
print(cmd)

client = paho.Client()
client.username_pw_set(ac.mqtt_login, ac.mqtt_pwd)
client.connect(host=ac.hostName)

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

PARAMETERS =  cv2.aruco.DetectorParameters_create()
MARKER_EDGE = 0.05


buffer = 64

# ac.redLower = (170, 106, 70)
# ac.redUpper = (185, 255, 255)

pts = deque(maxlen=buffer)


def sendMarkers(topic, msg):
	client.publish(topic, json.dumps(msg))

def angles_from_rvec(rvec):
    r_mat, _jacobian = cv2.Rodrigues(rvec)
    a = math.atan2(r_mat[2][1], r_mat[2][2])
    b = math.atan2(-r_mat[2][0], math.sqrt(math.pow(r_mat[2][1],2) + math.pow(r_mat[2][2],2)))
    c = math.atan2(r_mat[1][0], r_mat[0][0])
    return [a,b,c]

def calc_heading(rvec):
    angles = angles_from_rvec(rvec)
    degree_angle =  math.degrees(angles[2])
    if degree_angle < 0:
        degree_angle = 360 + degree_angle
    return degree_angle




def find_markers(frame, show=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, DICTIONARY, parameters=PARAMETERS)
    if show:
    	cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_EDGE, ac.CAMERA_MATRIX, ac.DIST_COEFFS)
    return corners, ids

def find_ball(frame, show=False):
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(hsv, ac.redLower, ac.redUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		if radius > 0 and show:
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	if show:
		pts.appendleft(center)
		for i in range(1, len(pts)):
			if pts[i - 1] is None or pts[i] is None:
				continue
			thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	if center != None:
		return x, y
		
	else:
		return -1, -1

def capture():

	os.system(cmd)

	if ac.showFrame:
		cv2.namedWindow("input")
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FPS, ac.framerate)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, ac.width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ac.height)
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
			time1 = time.time()
			frame_num += 1
			ret, img = cap.read()
			time2 = time.time()
			corners, ids = find_markers(img, ac.showFrame)
			jsonMarkers = """{
			"markers":[""" + "{},"*(len(corners)-1) + "{}" + """]
		}"""

			markers = json.loads(jsonMarkers)
			markers['count'] = len(corners)

			

			if len(corners) > 0:
				for i in range(0, len(corners)): #если найден хоть один маркер
					markers['markers'][i] = {'marker-id':int(ids[i][0]), 'camId' : ac.camId, 
							'corners': {'1':{'x':float(corners[i][0][0][0]),'y':float(corners[i][0][0][1])},
									'2':{'x':float(corners[i][0][1][0]),'y':float(corners[i][0][1][1])},
									'3':{'x':float(corners[i][0][2][0]),'y':float(corners[i][0][2][1])},
									'4':{'x':float(corners[i][0][3][0]),'y':float(corners[i][0][3][1])}
								}}
			sendMarkers(ac.topicRoot + ac.camId, markers)
			x, y = find_ball(img, ac.showFrame)
			
			if x != -1 and y != -1:
				ballData = {'ball':{'cam-id': ac.camId, 'center': {'x':float(x),'y':float(y)}}}
			else: 
				ballData = {'ball':'None'}
			sendMarkers(ac.topicBall + ac.camId, ballData)
			if ac.showFrame:
				cv2.imshow("input", img)
			key = cv2.waitKey(10)
			if key == 27:
				break

	except Exception as e:
		logging.error("Capturing stopped with an error:" + str(e))
	finally:
		cv2.destroyAllWindows()
		cv2.VideoCapture(0).release()

if __name__ == "__main__":
	capture()
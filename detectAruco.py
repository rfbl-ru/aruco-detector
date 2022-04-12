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

pts = deque(maxlen=buffer)


def sendMarkers(topic, msg):
	#publish.single(topic, json.dumps(msg), hostname=hostName, auth={'username' : mqtt_login, 'password': mqtt_pwd})
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

    # rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_EDGE, ac.CAMERA_MATRIX, ac.DIST_COEFFS)
    return corners, ids


kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(6, 6)) 
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

def find_ball(frame, corners, show=False):
	global kernel_close, kernel_open
	blank_image = np.zeros((ac.height,ac.width,1), np.uint8)
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
	circles = cv2.HoughCircles(mask.copy(), cv2.HOUGH_GRADIENT, 1, rows/8, param1=50, param2=7, minRadius=4, maxRadius=15)
	# cv2.imshow("mask", mask)
	if circles is not None:
		circles = np.uint16(np.around(circles))
		for circle in circles[0, :]:
			if show:
				cv2.circle(frame, (circle[0], circle[1]), 1, (0, 100, 100), 3)
				cv2.circle(frame, (circle[0], circle[1]), circle[2], (255, 0, 255), 3)
				
			return (circle[0], circle[1])
		
	return (-1, -1)	


def capture():

	os.system(cmd)

	if ac.showFrame:
		print("Show")
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
			frame_num += 1
			ret, img = cap.read()
			corners, ids = find_markers(img.copy(), ac.showFrame)
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
			center = find_ball(img, corners, ac.showFrame)
			if center[0] != -1:
				ball = {'ball':{'cam-id': ac.camId, 'center': {'x':float(center[0]),'y':float(center[1])}}}
			else:
				ball = {'ball':'None'}
			sendMarkers(ac.topicBall + ac.camId, ball)
			if ac.showFrame:
				cv2.imshow("input", img)
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

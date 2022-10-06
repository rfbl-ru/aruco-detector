import cv2
import numpy as np
from cameraFunctions import *
import arucoConfig as ac
import os
import sys

physCoords = [
	[10, 10], 		#0
	[185, 10],		#1
	[185, 283],		#2
	[10, 283],		#3
	[185, 146.5],	#4
	[10, 146.5],	#5
	[57, 62],		#6
	[138, 62],		#7
	[138, 231],		#8
	[57, 231]		#9
]

pitchCornersIds = [0, 1, 2, 3, 4, 5]

markerCoords = [
	[0, 0],
	[0, 0],
	[0, 0],
	[0, 0],
	[0, 0],
	[0, 0],
	[0, 0],
	[0, 0],
	[0, 0],
	[0, 0]
]

def calcPerspectiveMatrix(img):
	# cmd = "v4l2-ctl --set-ctrl=auto_exposure={0} --set-ctrl=exposure_time_absolute={1} --set-ctrl=brightness={2} " \
	#     "--set-ctrl=iso_sensitivity=1 "
	# cmd = cmd.format(ac.autoExposure, ac.exposureTime, ac.brightness)
	# os.system(cmd)
	# cv2.namedWindow("input")
	# cap = cv2.VideoCapture(0)
	# cap.set( cv2.CAP_PROP_FRAME_HEIGHT, ac.height )
	# cap.set( cv2.CAP_PROP_FRAME_WIDTH, ac.width )
	# cap.set(cv2.CAP_PROP_FPS, ac.framerate)

	# for i in range(100):
		# ret, img = cap.read()
	showImg = img.copy()

	corners, ids, showImg = find_markers(img, showImg, ac.showFrame)

	_i = 0

	if len(corners) > 0:
		for corner in corners:
			# print(ids[0])
			marker = calcArucoCenter(corner[0])
			markerCoords[ids[_i][0]] = [marker[0], marker[1]]
			_i += 1

	# print(markerCoords)
	newMarkerCords = []
	newPhysCoords = []

	for _id in pitchCornersIds:
		# print(markerCoords[i][0], markerCoords[i][1])
		if markerCoords[_id][0] != 0 and markerCoords[_id][1] != 0:
			newMarkerCords.append(markerCoords[_id])
			newPhysCoords.append(physCoords[_id])

	# print(newMarkerCords)
	# print(newPhysCoords)
	M = None
	isResult = False
	if len(newMarkerCords) == 4:
		M = cv2.getPerspectiveTransform(np.float32(newMarkerCords), np.float32(newPhysCoords))
		out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

		print(M)
		isResult = True
		cv2.imshow("out", out)
		
	return isResult, M
	cv2.imshow("input", showImg)
	# cv2.imwrite("input.jpg", showImg)
	cv2.waitKey(1)










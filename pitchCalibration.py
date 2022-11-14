import cv2
import numpy as np
from cameraFunctions import *
import arucoConfig as ac
import os
import sys

# Массив с физическими координатами калибровочных aruco меток
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
	showImg = img.copy()

	corners, ids, showImg = find_markers(img, showImg, ac.showFrame)

	_i = 0

	if len(corners) > 0:
		for corner in corners:
			marker = calcArucoCenter(corner[0])
			if ids[_i][0] < 6:
				markerCoords[ids[_i][0]] = [marker[0], marker[1]]
			_i += 1
	newMarkerCords = []
	newPhysCoords = []

	for _id in pitchCornersIds:
		if markerCoords[_id][0] != 0 and markerCoords[_id][1] != 0:
			newMarkerCords.append(markerCoords[_id])
			newPhysCoords.append(physCoords[_id])
	M = None
	isResult = False

	# В кадре должно быть только 4 калибровочные метки
	if len(newMarkerCords) == 4:
		M = cv2.getPerspectiveTransform(np.float32(newMarkerCords), np.float32(newPhysCoords))
		out = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

		print(M)
		isResult = True
		cv2.imshow("out", out)
		
	cv2.imshow("input", showImg)
	cv2.waitKey(1)
	return isResult, M, newMarkerCords

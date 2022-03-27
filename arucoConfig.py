import numpy as np


camId = "FootballCam-01"
topicRoot = "MIPT-SportRoboticsClub/LunokhodFootball/RawARUCO/"
topicBall = "MIPT-SportRoboticsClub/LunokhodFootball/RawBALL/"
hostName = "localhost"
mqtt_login = "explorer"
mqtt_pwd = "hnt67kl"
showFrame = True

width = 640
height = 480
framerate = 15

autoExposure = 1 # 0 - auto | 1 - manual
exposureTime = 40 # n * 1000 us
brightness = 50


# pi cam
CAMERA_MATRIX = np.array([[1481.7527738084875, 0.0, 936.8020968222914],[0.0, 1479.6348449552158, 557.8124212584266],[0.0, 0.0, 1.0]])
DIST_COEFFS = np.array([0.0006378238490344762,  2.4530717876176675,  0.00014402748184350837, -0.008451368578742634, -15.123459467944718])

redLower = (156, 80, 40)
redUpper = (190, 255, 255)


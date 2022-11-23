import numpy as np

#Настройки mqtt
camId = "2"
topicRoot = "MIPT-SportRoboticsClub/LunokhodFootball/RawARUCO/"
topicBall = "MIPT-SportRoboticsClub/LunokhodFootball/RawBALL/"
hostName = "localhost"
mqtt_login = "login"
mqtt_pwd = "pwd"
showFrame = True

#Настройки изображения с камеры
width = 640
height = 480
framerate = 20
linuxCameraNum = 0

autoExposure = 1 # 0 - auto | 1 - manual
exposureTime = 20 # n * 1000 us
brightness = 50


# pi cam
CAMERA_MATRIX = np.array([[1481.7527738084875, 0.0, 936.8020968222914],[0.0, 1479.6348449552158, 557.8124212584266],[0.0, 0.0, 1.0]])
DIST_COEFFS = np.array([0.0006378238490344762,  2.4530717876176675,  0.00014402748184350837, -0.008451368578742634, -15.123459467944718])

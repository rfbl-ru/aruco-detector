import cv2

#Candidate Blob detector

params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 10
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 20
params.maxArea = 90

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.7

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.5

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)


# ball blob detector

ballParams = cv2.SimpleBlobDetector_Params()

ballParams.minThreshold = 10
ballParams.maxThreshold = 200

# Filter by Area.
ballParams.filterByArea = True
ballParams.minArea = 5
# ballParams.maxArea = 90

# Filter by Circularity
ballParams.filterByCircularity = True
ballParams.minCircularity = 0.5

# Filter by Convexity
ballParams.filterByConvexity = False
ballParams.minConvexity = 0.5

# Filter by Inertia
ballParams.filterByInertia = False
ballParams.minInertiaRatio = 0.5

ballDetector = cv2.SimpleBlobDetector_create(ballParams)


#Aruco Markers params

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
PARAMETERS = cv2.aruco.DetectorParameters_create()


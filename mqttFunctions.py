import paho.mqtt.client as paho
import arucoConfig as ac
import json

client = paho.Client()
client.username_pw_set(ac.mqtt_login, ac.mqtt_pwd)
client.connect(host=ac.hostName)

def sendData(topic, data):
	client.publish(topic, json.dumps(data))

def sendMQTTData(ballData, corners, ids):

	jsonMarkers = """{
			"markers":[""" + "{}," * (len(corners) - 1) + "{}" + """],
			"camId" : """ + str(ac.camId) + "}"

	markers = json.loads(jsonMarkers)
	markers['count'] = len(corners)

	if len(corners) > 0:
		for i in range(0, len(corners)):
			markers['markers'][i] = {'marker-id': int(ids[i][0]), 'camId': ac.camId,
											 'corners': {'1': {'x': float(corners[i][0][0][0]),
															   'y': float(corners[i][0][0][1])},
														 '2': {'x': float(corners[i][0][1][0]),
															   'y': float(corners[i][0][1][1])},
														 '3': {'x': float(corners[i][0][2][0]),
															   'y': float(corners[i][0][2][1])},
														 '4': {'x': float(corners[i][0][3][0]),
															   'y': float(corners[i][0][3][1])}
														 }}
	sendData(ac.topicRoot + ac.camId, markers)                                                    
	if len(ballData) > 0:
		ball = {'camId': ac.camId, 'ball': []}
		for b in ballData:
			ball['ball'].append({'center': {'x': float(b[0]), 'y': float(b[1])}})
	else:
		ball = {'camId': ac.camId, 'ball': 'None'}
	sendData(ac.topicBall + ac.camId, ball)

def sendBallData(ballData):
	if len(ballData) > 0:
		ball = {'camId': ac.camId, 'ball': []}
		for b in ballData:
			ball['ball'].append({'center': {'x': float(b[0]), 'y': float(b[1])}})
	else:
		ball = {'camId': ac.camId, 'ball': 'None'}
	sendData(ac.topicBall + ac.camId, ball)
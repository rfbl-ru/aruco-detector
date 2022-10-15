import paho.mqtt.client as paho
import json
import time

cam1Data = []
cam2Data = []

def onMessage(client, userdata, msg):
	data = json.loads(msg.payload.decode('utf-8'))
	if data['camId'] == "1":
		cam1Data.append([time.time(), data['ball']])
	elif data['camId'] == "2":
		cam2Data.append([time.time(), data['ball']])


client = paho.Client()
client.on_message = onMessage
client.username_pw_set("calliska", "mXCRI5")
client.connect(host="192.168.3.104")

client.subscribe("MIPT-SportRoboticsClub/LunokhodFootball/RawBALL/#")
try:
	client.loop_forever()
except KeyboardInterrupt:
	with open('cam1.json', 'w', encoding='utf-8') as f:
		json.dump(cam1Data, f)
	with open('cam2.json', 'w', encoding='utf-8') as f:
		json.dump(cam2Data, f)

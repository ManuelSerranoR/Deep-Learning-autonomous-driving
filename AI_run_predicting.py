import socket
import sys
import math
import pygame
from pygame.locals import *
import os
import time
import mss
import mss.tools
from random import randint
import numpy as np
from PIL import Image
import pickle
import pyautogui
from skimage import transform #For downsizing images
import tensorflow as tf
sys.path.append('/home/canlab/Desktop/gym_torcs')
from snakeoil3_gym_old import *

tf.GraphKeys.USEFUL = 'useful'
mean_image = [86.49175515, 91.82099339, 88.3039215]

#Controllers
K11 = [9.23367926763609,    1.95275841037166,    2.72676657363386,    -0.242125733895282]
K12 = [12.6977207762501,    2.74374094090788,    2.96618342747267,    -0.591707929386049]
K13 = [2.79995871454660,    0.437718377480784,   -0.181819647702710,  0.000902734259494991]
K14 = [17.3952997585746,    3.89650745470490,    6.30719267080744,    -0.509216683749110]
K15 = [42.1935979235132,    9.76400687139357,    17.4133997845239,    -1.75004024168088]
K16 = [14.6525527110655,    3.24460508533655,    5.06178291138374,    -0.531858811257178]
K17 = [49.9659836483932,    11.6156203022342,    20.8389206311191,    -2.02144735902972]
K18 = [4.88926526279235,    0.941812964337513,   0.712483291441938,   -0.0427015511972851]
K19 = [13.5931670099812,    2.89804410735146,    4.12451826819267,    -0.495894907656895]
K20 = [10.0958121862069,    2.14046234200640,    2.91703565069511,    -0.270506090822326]

#With this class we are able to import multiple graphs using different instances
class ImportGraph():		    
	"""  Importing and running isolated TF graph """		    
	def __init__(self, loc):
		# Create local graph and use it in the session
		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph)
		with self.graph.as_default():
			# Import saved model from location 'loc' into local graph
			saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
			saver.restore(self.sess, loc)
			# Get activation function from saved collection
			# You may need to change this in case you name it differently
			self.var_list = tf.get_collection(tf.GraphKeys.USEFUL)

	def predict(self, data):
		""" Running the activation function previously imported """
		# The 'x' corresponds to name of input placeholder
		#The contents of var_list depend on how this collection was defined when saving the model
		# In this case, var_list[3] corresponds to the predicted value of the net, and var_list[0] to the sample given to be predicted
		return self.sess.run(self.var_list[3], feed_dict={self.var_list[0]:[data]})


#Code for the driving parameters window
pygame.init()
DISPLAYSURF = pygame.display.set_mode((700, 300))
pygame.display.set_caption('Driving parameters')


def drive(c, time_diff, angle_diff, trackPos_diff):
	'''This is only an example. It will get around the track but the
	correct thing to do is write your own `drive()` function.'''
	S,R = c.S.d, c.R.d
	print(S['speedX'])
	if S['speedX'] > 10:
		x = [S['trackPos'], trackPos_diff/time_diff, S['angle'], angle_diff/time_diff]
		R['steer'] = - (np.dot(K12, x))
	else:
		R['steer'] = 0.7*S['angle'] - S['trackPos']

	R['accel'] = .15
	if S['speedX'] > 53:
		R['brake'] = .1
	else: R['brake'] = .0

# ================ MAIN ================
if __name__ == "__main__":
	C= Client(p=3101)
	time_new = 0.2
	time_old = 0
	angle_new = 0
	angle_old = 0
	trackPos_new = 0
	trackPos_old = 0

	#Load both models
	model_trackPos = ImportGraph('../train/CNN/model_for_trackPos_prediction/my_test_model')
	model_angle = ImportGraph('../train/CNN/model_for_angle_prediction/my_test_model')
	
	for step in range(C.maxSteps):
		print('Step',step,'out of',C.maxSteps-1)
		if step == 0:
			print("--------------")
			print("change view")
			print("--------------")
			pyautogui.press("f2")

		C.get_servers_input()
		
		# Capture Image
		with mss.mss() as sct:
			monitor = {'top': 52, 'left': 65, 'width': 640, 'height': 480}
			# Grab the data
			sct_img = sct.grab(monitor)
		img = Image.frombytes('RGBA', sct_img.size, bytes(sct_img.raw), 'raw', 'BGRA')
		img = np.asarray(img.convert('RGB').resize((224,224))) - mean_image

		#Prediction from the trained models we just loaded
		predicted_trackPos = model_trackPos.predict(img)
		predicted_angle = model_angle.predict(img)

		#print('Current real output: ', C.S.d['angle']*(180/math.pi))
		print('Current real trackPos: ', C.S.d['trackPos'])
		print('Current predicted trackPos: ', predicted_trackPos)

		print('Current real angle: ', C.S.d['angle'])
		print('Current predicted angle: ', predicted_angle)


		#Code for the driving parameters window: First we set the screen back to black
		DISPLAYSURF.fill((0, 0, 0))

		#Now we upgrade the driving parameters
		DP_angle_truth_x = 200 + math.sin(C.S.d['angle'])*150
		DP_angle_truth_y = 200 - math.cos(C.S.d['angle'])*150
		DP_angle_predicted_x = 200 + math.sin(predicted_angle[0][0]*(math.pi/180))*150
		DP_angle_predicted_y = 200 - math.cos(predicted_angle[0][0]*(math.pi/180))*150

		DP_trackPos_truth = C.S.d['trackPos']
		DP_trackPos_predicted = predicted_trackPos[0][0]/8

		#Draw
		pygame.draw.line(DISPLAYSURF, (255, 255, 255), (200, 250), (200,50), 1)
		pygame.draw.aaline(DISPLAYSURF, (141, 236, 120), (200, 250), (DP_angle_truth_x, DP_angle_truth_y), 3)
		pygame.draw.aaline(DISPLAYSURF, (199, 44, 58), (200, 250), (DP_angle_predicted_x,DP_angle_predicted_y), 3)

		pygame.draw.line(DISPLAYSURF, (255, 255, 255), (500, 190), (500,260), 1)
		pygame.draw.aaline(DISPLAYSURF, (141, 236, 120), (500, 250), (500 + DP_trackPos_truth*400,250), 3)
		pygame.draw.aaline(DISPLAYSURF, (199, 44, 58), (500, 200), (500 + DP_trackPos_predicted*400,200), 3)
		pygame.display.update()


		time_old = time_new
		time_new = time.time()
		time_diff = time_new - time_old

		print('TIME:', time_diff)

		trackPos_old = trackPos_new
		trackPos_new = C.S.d['trackPos']
		trackPos_diff = trackPos_new - trackPos_old

		print('DIST:', trackPos_diff)

		angle_old = angle_new
		angle_new = C.S.d['angle']
		angle_diff = angle_new - angle_old

		print('ANGLE:', angle_diff)

		drive(C, time_diff, angle_diff, trackPos_diff)
		C.respond_to_server()


	print("--------------")
	print("Close window")
	print("--------------")

	sys.exit()
	C.shutdown()
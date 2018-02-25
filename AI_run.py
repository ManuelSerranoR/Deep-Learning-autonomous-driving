import socket
import sys
import getopt
import math
import os
import cv2
import time
import mss
import mss.tools
import random
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import pyautogui
from skimage import transform #For downsizing images
import tensorflow as tf
sys.path.append('/home/canlab/Desktop/gym_torcs')
from snakeoil3_gym_old import *


K1 = [10.9908717467978, 0.195270206285503, 2.72666097213649, -0.0242112941279921]
K2 = [29.3088785103195, 0.554720938680623, 9.32215050455478, -0.0783918237579237]
K3 = [15.1634806408263, 0.274302220134651, 2.96487374439337, -0.0591630761603098]
K4 = [20.9021653759770, 0.389650930651144, 6.30719447180785, -0.0509215760266584]
K5 = [50.9714030885695, 0.976208001341686, 17.4099031409000, -0.174972239077307]
K6 = [20.1472107264722, 0.374579805036356, 5.99662162767896, -0.0618876680162685]
K7 = [50.8841442241103, 0.974420194231805, 17.2845730533334, -0.168859625168260]
K8 = [5.74658671795202, 0.0943574176842459, 0.715637550153061, -0.00431826756492982]
K9 = [80.4708063923528, 1.55708090168432, 27.4692802112351, -0.262353879617020]
K10 = [112.018020379178, 2.15364021346955, 37.8405518624785, -0.427361896649282]


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

    C = Client(p = 3101)
    time_new = 0.2
    time_old = 0
    angle_new = 0
    angle_old = 0
    trackPos_new = 0
    trackPos_old = 0


    for step in range(C.maxSteps):
        print('Step',step,'out of',C.maxSteps-1)
        if step == 0:
            print("--------------")
            print("change view")
            print("--------------")
            pyautogui.press("f2")

        #This causes the delay of 0.2 seconds
        C.get_servers_input()

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

    pyautogui.click(80, 40, button='left')
    sys.exit()

    C.shutdown()
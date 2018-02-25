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


def drive(c):
    '''This is only an example. It will get around the track but the
    correct thing to do is write your own `drive()` function.'''
    S,R= c.S.d,c.R.d

    if S['trackPos'] < -0.17 or S['trackPos'] > 0.155:
        R['steer'] = 1.15*(S['angle'] - S['trackPos'])
    R['accel'] = .2
    R['brake'] = .0

    return R['steer']

# ================ MAIN ================
if __name__ == "__main__":
    train_data = []
    obs = []
    C= Client(p=3101)

    index_image = []
    speedX = []
    speedY = []
    angle = []
    radius = []
    trackPos = []
    seg_width = []
    steer = []

    for step in range(C.maxSteps):
        print('Step',step,'out of',C.maxSteps-1)
        if step == 0:
            print("--------------")
            print("change view")
            print("--------------")
            pyautogui.press("f2")
            pyautogui.press("4")

        C.get_servers_input()


        steer_return = drive(C)

        
        # Capture Image
        with mss.mss() as sct:
            monitor = {'top': 52, 'left': 65, 'width': 640, 'height': 480}
            # Grab the data
            sct_img = sct.grab(monitor)
        img = Image.frombytes('RGBA', sct_img.size, bytes(sct_img.raw), 'raw', 'BGRA')
        img = img.convert('RGB').resize((224,224))
        img.save('dataX/'+str(step)+'.png')
        

        #print(C.S.d['speedX'], C.S.d['speedY'], C.S.d['angle'], C.S.d['radius'], C.S.d['trackPos'], C.S.d['seg_width'], steer_return)

        index_image.append(step)
        speedX.append(C.S.d['speedX'])
        speedY.append(C.S.d['speedY'])
        angle.append(C.S.d['angle'])
        radius.append(C.S.d['radius'])
        trackPos.append(C.S.d['trackPos'])
        seg_width.append(C.S.d['seg_width'])
        steer.append(steer_return)
        
        C.respond_to_server()

    collected_Data_Mengzhe = pd.DataFrame(np.array([np.array(index_image),
                                                    np.array(speedX), 
                                                    np.array(speedY), 
                                                    np.array(angle), 
                                                    np.array(radius), 
                                                    np.array(trackPos),
                                                    np.array(seg_width),
                                                    np.array(steer)]).transpose(), 
                                                    columns=["index_image", "speedX", "speedY", "angle", "radius", "trackPos", "seg_width", "steer"])

    collected_Data_Mengzhe.to_csv("dataY.csv")

    print("--------------")
    print("Close window")
    print("--------------")

    pyautogui.click(80, 40, button='left')
    sys.exit()

    C.shutdown()
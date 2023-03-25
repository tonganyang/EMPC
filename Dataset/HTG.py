# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:48:47 2022

@author: TAY
"""


# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

folder = "/home/hfuu/TAY/Supervise/3D/data/kinetic100"
interval = 6

def Histogram(frame):
    
    Imin, Imax = cv2.minMaxLoc(frame)[:2]
    Omin, Omax = 0, 255
    if (Imax - Imin) == 0:
        histogram = frame.astype(np.float32)
    else:
        ImaxImin = (Imax - Imin)
        a = float(Omax - Omin) / ImaxImin
        b = Omin - a * Imin
        histogram = a * frame + b
        histogram = histogram.astype(np.float32)
        
    return histogram

def TGradient(imglist):
    
    length = len(imglist)
    now = (length-1)*(imglist[length-1].astype(np.float32))
    his = imglist[0].astype(np.float32)
    for i in range(length-2):
        his = his + imglist[i+1].astype(np.float32)
    temp_grad = Histogram(now - his)
    temp_grad += 180.0
    temp_grad /= 2
    temp_grad = temp_grad.astype(np.uint8)
    
    return temp_grad

for file in sorted(os.listdir(folder)):
    if file == "test(RGB)":
        path_to_RGB = os.path.join(folder, file)
        path_to_HTG = os.path.join(folder, "test+(HTG)")
        
        for label in sorted(os.listdir(os.path.join(folder, file))):
            
            label_to_RGBdata = os.path.join(path_to_RGB, label)
            label_to_HTGdata = os.path.join(path_to_HTG, label)
            
            if not os.path.exists(label_to_HTGdata): #  创建文件夹
                os.mkdir(label_to_HTGdata)
                
            for video in sorted(os.listdir(label_to_RGBdata)):
                video_to_rgbimg = os.path.join(label_to_RGBdata, video)
                video_to_htgimg = os.path.join(label_to_HTGdata, video)
                
                for i in range(len(sorted(os.listdir(video_to_rgbimg)))-interval):
                    imglist = []
                    for j in range(interval):
                        RGB_img_path = os.path.join(video_to_rgbimg, '0000{}.jpg'.format(str(i+j)))
                        img = cv2.imread(RGB_img_path)
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        imglist.append(img)
                    HTG = TGradient(imglist)
                    if not os.path.exists(video_to_htgimg): #  创建文件夹
                        os.mkdir(video_to_htgimg)
                    cv2.imwrite(filename=os.path.join(video_to_htgimg, '0000{}.jpg'.format(str(i))), img=HTG)
                    
    else:
        continue


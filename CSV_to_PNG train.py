import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt
import csv
import png
import cv2
'''
HOW TO:
1- Choose the folder destination for lungs (destlung) and not lungs (destnot). These MUST be subfolders of the same folder.
1.a - If you're using Windows use \\ for path steps
2- Choose pixels.csv (pixelfile) and labels.csv (labelfile)
3- Let the magic happen
'''
destlung ='D:\\Emanuele\\Progetto\\Proverino\\piennegi\\lungs'
destnot ='D:\\Emanuele\\Progetto\\Proverino\\piennegi\\notlungs'

with open ('D:\Emanuele\Progetto\Proverino\minidata.csv', newline='') as pixelfile:
    pixel_reader = csv.reader(pixelfile)
    pixel_data = []
    for row in pixel_reader:
        pixel_data.append(row)

pixels = np.array(pixel_data)
#pixels = pixels.astype(float)
#pixel_data = pixel_data.astype(float)
#resized = []
with open ('D:\Emanuele\Progetto\Proverino\minilabel.csv', newline='') as labelfile:
    label_reader = csv.reader(labelfile)
    label_data = []
    for row in label_reader:
        label_data.append(row)

labels = np.array(label_data)
labels = labels.astype(int)
lungcount=0
notcount=0
for i in range(len(pixels)):
    pix = pixel_data[i]
    lab = labels[i]
    #pix = pix.astype(float)
    pix = np.asarray(pix)
    pix= pix.astype(float)
    pix = np.reshape(pix,(-1,128))
    res =cv2.resize(pix, (140,140), interpolation = cv2.INTER_LINEAR)
    res = (np.maximum(res,0) / res.max()) * 255.0
    res = np.uint8(res)
    #resized.append(res)
    if lab==1:
        lungcount+=1
        with open((destlung + "\lungs" + "%s" %lungcount +".png"), 'wb') as png_file:
            w = png.Writer(140, 140, greyscale=True)
            w.write(png_file, res)
    else:
        notcount+=1
        with open((destnot + "\\not" + "%s" %notcount +".png"), 'wb') as png_file:
            w = png.Writer(140, 140, greyscale=True)
            w.write(png_file, res)

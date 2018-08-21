import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt
import csv
import png
import cv2

destination="/Users/utente/Desktop/Test"

with open ('/Users/utente/Desktop/ND_CSV/pixel_data.csv', newline='') as pixelfile:
    pixel_reader = csv.reader(pixelfile)
    pixel_data = []
    for row in pixel_reader:
        pixel_data.append(row)

pixels = np.array(pixel_data)
#pixels = pixels.astype(float)
#pixel_data = pixel_data.astype(float)
#resized = []

for i in range(len(pixels)):
    pix = pixel_data[i]
    #pix = pix.astype(float)
    pix = np.asarray(pix)
    pix= pix.astype(float)
    pix = np.reshape(pix,(-1,128))
    res =cv2.resize(pix, (140,140), interpolation = cv2.INTER_LINEAR)
    res = (np.maximum(res,0) / res.max()) * 255.0
    res = np.uint8(res)
    #resized.append(res)
    with open((destination + "/test" + "%s" %i +".png"), 'wb') as png_file:
        w = png.Writer(140, 140, greyscale=True)
        w.write(png_file, res)

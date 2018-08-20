#Convert .dcm files to .png files
import os
import sys
import numpy as np
import png
import pydicom
import cv2

#TODO define path and destination according to user 

os.chdir("/Users/utente/Test")
path="/Users/utente/Test/"
destination="/Users/utente/Desktop/Test"

for filename in os.listdir(path):

	if filename.endswith(".dcm"):

		ds = pydicom.dcmread(filename)
		new_data = ds.pixel_array
		#print(new_data.shape)
        
        #TODO fix this step 
		#if ds.HighBit == 15:
		#	pad_val = ds.PixelPaddingValue #& 0xffff
		#	for i in range(0,512):
		#		if any(new_data[i] == pad_val) or any(new_data[i] == -((pad_val-1) ^ 0xffff)): #(new_data[i] > (2^15-1))
		#			new_data[i] = 0

		# Convert to float to avoid overflow or underflow losses.
		new_data = new_data.astype(float)

		new_data = np.reshape(new_data, (-1,512))
		#width, height =new_data.shape[:4]
		resized=cv2.resize(new_data, (128,128), interpolation = cv2.INTER_LINEAR) #better LINEAR for downsampling 
		#np.reshape was used for .csv, won't work here
		#resized = np.reshape(resized, (16384))

		# Rescaling grey scale between 0-255
		resized = (np.maximum(resized,0) / resized.max()) * 255.0

		# Convert to uint
		resized = np.uint8(resized)

		# Write the PNG file
		with open((os.path.join(destination,filename.strip(".dcm"))+".png"), 'wb') as png_file:
		    w = png.Writer(128, 128, greyscale=True)
		    w.write(png_file, resized)

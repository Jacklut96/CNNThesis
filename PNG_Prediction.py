import pandas as pd
import csv
import os
import re
import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import np_utils
import numpy as np 
#import matplotlib.pyplot as plt

path ="/home/users/emanuele.tauro/Desktop/totale/"
n_samples = 0 
good =  0
fn = 0 
fp = 0

#LABEL READING (Future work ------> FILENAME CHECK)
#with open ('/Users/utente/Desktop/ND_CSV/labels.csv') as labelfile:
#	label_reader = csv.reader(labelfile)
#	label_data = []
#	for row in label_reader:
#		label_data.append(row)


#LABEL_AS_ARRAY (To be implemented)
#label_array = []
#for i in range (0,488):
#    label_array.append(np.asarray(label_data[i], float))
    
#label_array = np.reshape(label_array,(488,1))
#label_array = label_array.astype("Float32")
        


#-------MODEL LOADING-----------------
model = load_model('/home/users/emanuele.tauro/Desktop/models/Inceptionfullr42.h5')
img_width = 140
img_height = 140 

#-------PREDICTION AND RESULT--------------------
for filename in os.listdir(path):
    if filename.endswith(".png"):
        n_samples += 1
        img = image.load_img(os.path.join(path,filename), target_size=(img_width, img_height)) 

        # convert image to numpy array, so Keras can render a prediction
        img_arr = image.img_to_array(img)
        #img_arr = (np.maximum(img_arr,0) / img_arr.max())

        # expand array from 3 dimensions (height, width, channels) to 4 dimensions (batch size, height, width, channels)
        # rescale pixel values to 0-1
        x = np.expand_dims(img_arr, axis=0) * 1./255

        # get prediction on test image 
        score = model.predict(x)
        text = ("%s" %score + ": lung" if score < 0.5 else "%s" %score + ": not lung")
        #append predictions to have quantitative results
    
        prediction = 1 if score <0.5 else 0
        
        #if (score<=0.5):
        #    prediction = 1
        #if (score>0.5):
        #    prediction = 0

        if re.match("^lung", filename):
            truth = 1
        if re.match("^not", filename):
            truth = 0

        if (prediction==truth):
            good += 1

        if(prediction==1 and truth==0):
            fp += 1
        if(prediction==0 and truth==1):
            fn += 1

        #==Plotting (can be commented safely)====
        #plt.imshow(img)
        #plt.text(0,(-2.5),text)
        #plt.show()


accuracy = (good/n_samples)*100
print("Accuracy: %f" %accuracy)
print("False positives: %i" %fp)
print("False negatives: %i" %fn)

#----------PREDICTION RESULT---------------- (TODO: implement label check)
#result = model.predict(pixel_array)
#prediction = result[:,1]
#count=0 
#error = 0

#for i in range(0, 488):
#    if label_array[i] > 0 and label_array[i] < 1:
#        label_array[i]=0
#    if prediction[i] > 0 and prediction[i]<1:
#        prediction[i]=0
#        
#    if label_array[i]==prediction[i]:
#        count +=1
#    else:
#        error +=1

#acc = count/488
#print (acc*100)
#predlist = prediction.tolist()
#predlist = [int(i) for i in predlist]
#print(elapsed_time = time.time() - start)

#for i in range (0,100):
#    pixel_data[i] = list(map(int, pixel_data[i]))
#    pixel_data[i] = np.reshape(pixel_data[i],(-1,128))
#    plt.imshow(pixel_data[i], cmap="gray")
#    plt.title(predlist[i])
#    plt.show()

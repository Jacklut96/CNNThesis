import os
import random
import shutil

random.seed(42)
datanum = 49860 #dataset dimension

size=int(datanum/2)
validation=int(size*0.2)

src_lun = "/home/users/emanuele.tauro/Desktop/PNG/lungs"
train_lun = "/home/users/emanuele.tauro/Desktop/train/lungs"
src_not = "/home/users/emanuele.tauro/Desktop/PNG/notlungs"
train_not = "/home/users/emanuele.tauro/Desktop/train/notlungs"
val_lun = "/home/users/emanuele.tauro/Desktop/validation/lungs"
val_not = "/home/users/emanuele.tauro/Desktop/validation/notlungs"


pick= random.sample(os.listdir(src_lun), size) #insert source
for i in range(len(pick)):
	shutil.copy(src_lun + "/" + pick[i], train_lun) #insert destination

pick= random.sample(os.listdir(src_not), size) #insert source
for i in range(len(pick)):
	shutil.copy(src_not + "/" + pick[i], train_not) #insert destination

pick= random.sample(os.listdir(train_lun), validation) #insert source
for i in range(len(pick)):
	shutil.move(train_lun + "/" + pick[i], val_lun) #insert destination

pick= random.sample(os.listdir(train_not), validation) #insert source
for i in range(len(pick)):
	shutil.move(train_not + "/" + pick[i], val_not) #insert destination

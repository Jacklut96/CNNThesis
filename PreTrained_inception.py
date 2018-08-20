 
# load requirements from the Keras library
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D 
from keras.models import Model
from keras.optimizers import Adam

# dimensions of our images  (TODO: change the dimension of the images)
img_width, img_height = 299, 299

# directory and image information (TODO: change the path to the files)
train_data_dir = '../data/train' 
validation_data_dir = '../data/val'

# epochs = number of passes of through training data 
#batch_size = number images processed at same time 
#TODO: change these dimensions

train_samples = 65
validation_samples = 10
epochs = 20
batch_size = 5

#--------DATASET INITIALIZATION--------
#TODO: add data transformation in the train_datagen to improve accuracy
train_datagen = ImageDataGenerator(
rescale= 1./255)
val_datagen = ImageDataGenerator(
rescale=1./255)

# Directory, image size, batch size already specified above
# Class mode is set to 'binary' for a 2-class problem
# Generator randomly shuffles and presents images in batches to the network

train_generator = train_datagen.flow_from_directory( train_data_dir,
target_size=(img_height, img_width), batch_size=batch_size,
class_mode='binary')
validation_generator = train_datagen.flow_from_directory( validation_data_dir,
target_size=(img_height, img_width), batch_size=batch_size,
class_mode='binary')

#--------MODEL INITIALIZATION---------
# build the Inception V3 network, use pretrained weights from ImageNet # remove top fully connected layers by include_top=False
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# build a classifier model to put on top of the convolutional model
# This consists of a global average pooling layer and a fully connected layer with 256 nodes # Then apply dropout and sigmoid activation
model_top = Sequential() model_top.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:], data_format=None)),
model_top.add(Dense(256, activation='relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(1, activation='sigmoid'))
model = Model(inputs=base_model.input, outputs=model_top(base_model.output))
# Compile model using Adam optimizer with common values and binary cross entropy loss # Use low learning rate (lr) for transfer learning
model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e- 08,decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])

#---------TRAINING------------------
history = model.fit_generator( train_generator,
steps_per_epoch=train_samples // batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=validation_samples // batch_size)

#---------EVALUATION----------------

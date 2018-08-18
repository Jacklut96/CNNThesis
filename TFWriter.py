import argparse
import os
import sys
import csv
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import islice


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#apri il file del .csv da convertire
pf = open('/Users/utente/Desktop/LilDataset/pixel_data.csv', 'r') #cambiando l'apertura file per contare gli elementi nel .csv , non va bene (nessuna ridondanza quindi)

pixelfile = '/Users/utente/Desktop/LilDataset/pixel_data.csv' #filepath da usare per pandas per estrarre i valori numerici che non danno problemi
labelfile = '/Users/utente/Desktop/LilDataset/labels.csv'

#inizializzo i lettori di .csv
pixelreader = csv.reader(pixelfile)
labelreader = csv.reader(labelfile)

leng = []
#conto le immmagini nel .csv in esame
for row in pf:
    leng.append(0)


#stabilisco la ripartizione del dataset in TRAINING/TEST/EVALUATION
tr_l = math.floor(len(leng) / 10) * 6
te_l = math.floor(len(leng) / 10) * 2
ev_l = len(leng) - tr_l - te_l

#TRAINING
train_filename = '/Users/utente/Desktop/train_filename.tfrecords'  #dove salvare il file .tfrecords di training
test_writer = tf.python_io.TFRecordWriter(train_filename) #apro il writer relativo
#scrivo riga per riga nel .tfrecord
for i in range(1,tr_l):
    img = pd.read_csv(pixelfile, header=None, nrows=i, delim_whitespace=True)
    img = img.at[0,0]
    img = img.split(',')
    img = list(map(float, img))
    img = np.reshape(img, (128,128))

    l = pd.read_csv(labelfile, header=None, nrows=i, delim_whitespace=True)
    label = l.at[0,0]

    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    test_writer.write(example.SerializeToString())
    print("%d \n", i)

test_writer.close() #chiudo il writer



#VALUTAZIONE
val_filename = '/Users/utente/Desktop/val_filename.tfrecords'  #dove salvare il .tfrecords di valutazione
val_writer = tf.python_io.TFRecordWriter(val_filename)  #apro il writer

for i in range(tr_l + 1, tr_l + ev_l):
    img = pd.read_csv(pixelfile, header=None, nrows=i, delim_whitespace=True)
    img = img.at[0,0]
    img = img.split(',')
    img = list(map(float, img))
    img = np.reshape(img, (128,128))

    l = pd.read_csv(labelfile, header=None, nrows=i, delim_whitespace=True)
    label = l.at[0,0]

    feature = {'val/label': _int64_feature(label),
               'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    val_writer.write(example.SerializeToString())
    print("%d \n", i)

val_writer.close() #chiudo il writer relativo alla valutazione


#TEST
test_filename = '/Users/utente/Desktop/test_filename.tfrecords'  #dove salvare il .tfrecords di test
test_writer = tf.python_io.TFRecordWriter(test_filename)  #apro il writer

for i in range(tr_l + ev_l + 1, len(leng)):
    img = pd.read_csv(pixelfile, header=None, nrows=i, delim_whitespace=True)
    img = img.at[0,0]
    img = img.split(',')
    img = list(map(float, img))
    img = np.reshape(img, (128,128))

    l = pd.read_csv(labelfile, header=None, nrows=i, delim_whitespace=True)
    label = l.at[0,0]

    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    test_writer.write(example.SerializeToString())
    print("%d \n", i)

test_writer.close() #chiudo il writer relativo al test






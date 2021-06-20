# -*- coding: utf-8 -*-
"""
# <center> Animals Images Classification

---

<center> [dataset](https://www.kaggle.com/antoreepjana/animals-detection-images-dataset)<br><small> *note: the output was run on GPU mode*
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.image import imread
from pip._internal import main as pipmain
pipmain(['install', 'tensorflow-addons'])

from tensorflow.keras import Sequential, Input
from tensorflow.keras.applications import ResNet50V2, InceptionV3
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow_addons.optimizers import AdamW
tf.random.set_seed(233)
np.random.seed(233)

# Define the directory
data_dir = '../input/animals-detection-images-dataset/train'

# Check the sub-folders that show all classes
anml_types = np.array(os.listdir(data_dir))

# Count each class number
ty_tmp = []
for abc in anml_types:
    ty_tmp.append(len(os.listdir(data_dir + '/' + abc))-1) # minus the directories folder

cnt_df = pd.DataFrame(anml_types,ty_tmp).reset_index().sort_values(by='index', ascending=False).reset_index(drop=True)
cnt_df.columns = ['counts', 'animal']

# just using these 48 classes (100-700 image file each)
filter_img = cnt_df[(cnt_df.counts>100) & (cnt_df.counts<700)]

"""# Data Preprocessing"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.random.set_seed(233)

train_gen = ImageDataGenerator(rescale=1/255,
                                rotation_range=35,
                                zoom_range=.1,
                                horizontal_flip=True,
                                validation_split=.2,
                                )

valid_gen = ImageDataGenerator(rescale=1/255,
                                validation_split=.2,
                                )

# File to modelling
IMG_SHAPE = (400,500,3)
BATCH_SIZE = 16
SEED = 233
CLASS_FLT = list(filter_img.animal.values)

train_img_gen = train_gen.flow_from_directory(data_dir,
                                            target_size=IMG_SHAPE[:2],
                                            batch_size=BATCH_SIZE,
                                            seed=SEED,
                                            shuffle=True,
                                            classes=CLASS_FLT,
                                            class_mode='sparse',
                                            subset='training') 

valid_img_gen = valid_gen.flow_from_directory(data_dir, 
                                            target_size=IMG_SHAPE[:2],
                                            batch_size=BATCH_SIZE,
                                            seed=SEED,
                                            shuffle=False,
                                            classes=CLASS_FLT,
                                            class_mode='sparse',
                                            subset='validation')

# pd.DataFrame([train_img_gen.class_indices, valid_img_gen.class_indices], ['train', 'valid'])

"""# Modelling"""
SCHEDULE = tf.optimizers.schedules.PiecewiseConstantDecay([1407*20, 1407*30], [1e-3, 1e-4, 1e-5])
step = tf.Variable(0, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay([10000, 15000], [1e-0, 1e-1, 1e-2])
LR = 1e-1 * schedule(step)
WD = lambda: 1e-4 * SCHEDULE(step)
OPTIMIZER = AdamW(learning_rate=SCHEDULE, weight_decay=WD)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

class custom_callback(Callback):
    total_t0 = 0
    def on_train_begin(self, logs={}):
        self.total_t0 = time.time()
    def on_train_end(self, logs={}):
        print('')
        print('Training complete!')
        print('Total training took {:} (h:mm:ss)'.format(format_time(time.time()-self.total_t0)))
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.92):
            print("\nReached >92% accuracy so stopping training!")
            self.model.stop_training = True
        
def build_model():
    # Transfer Learning with custom output
    base_model = ResNet50V2(input_shape=(202,252,3), include_top=False)
    base_model.trainable = False
#   base_model.layers[5:]
    
    model = Sequential()
    model.add(Input(shape=IMG_SHAPE))
    model.add(ZeroPadding2D())
#     model.add(Dropout(.2))
    model.add(Conv2D(3, 3, padding='same', activation='relu'))
    model.add(ZeroPadding2D())
#     model.add(Dropout(.2))
    model.add(MaxPool2D(2, 2))
    model.add(BatchNormalization())

    model.add(base_model)
    
    model.add(GlobalAveragePooling2D())
#     model.add(Dense(1024,activation='relu'))
#     model.add(Dense(512,activation='relu'))
#     model.add(Dense(256,activation='relu'))
#     model.add(Dense(128,activation='relu'))
#     model.add(Dropout(.2))
#     model.add(Dense(96,activation='relu'))
#     model.add(Dropout(.2))
#     # Output layer
    model.add(Dense(48, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    return model

callbacks = custom_callback()
stopper = EarlyStopping(patience=74, min_delta=0.05, baseline=0.8,
                        mode='min', monitor='accuracy', 
                        restore_best_weights=True, verbose=1)

model = build_model()
tf.keras.utils.plot_model(model, show_shapes=True, rankdir='TP')

hist = model.fit(train_img_gen, 
                epochs=76,
                steps_per_epoch=5,
                validation_steps=5,
                validation_data=valid_img_gen, 
                callbacks=[stopper, callbacks], 
                verbose=2)

"""# Evaluation"""

eval_df = pd.DataFrame(hist.history)
length = len(eval_df)

"""# Save Model for Deployment"""

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with tf.io.gfile.GFile('model_animal.tflite', 'wb') as f:
    f.write(tflite_model)
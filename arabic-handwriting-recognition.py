
# coding: utf-8

# In[1]:


import os
import cv2
import pandas as pd
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PIXELS = 1024
DIMENSIONS = np.int16(math.sqrt(PIXELS))

TRAINING_FEATURES_FILE = "csvTrainImages 13440x1024.csv"
TRAINING_LABELS_FILE = "csvTrainLabel 13440x1.csv"
TESTING_FEATURES_FILE = "csvTestImages 3360x1024.csv"
TESTING_LABELS_FILE = "csvTestLabel 3360x1.csv"
MODEL_NAME = "trained_model.h5"

PCA_THRESHOLD = 0.9

def load_data(file=TRAINING_FEATURES_FILE, header=True):
    csv_path = os.path.join("dataset/", file)
    if header:
        return pd.read_csv(csv_path)
    else:
        return pd.read_csv(csv_path, header=None)


# In[2]:


data = load_data(TRAINING_FEATURES_FILE)
data.head()


# In[3]:


def imagify(arr, getimage=False):
    img = np.array(np.reshape(arr, (DIMENSIONS, DIMENSIONS)), dtype="uint8")

    if getimage:
        return img


# In[4]:


THRESH_BINARY = cv2.THRESH_BINARY
THRESH_BINARY_AND_THRESH_OTSU = cv2.THRESH_BINARY+cv2.THRESH_OTSU


# In[5]:


def apply_thresholding(df, cap=0, thres=THRESH_BINARY_AND_THRESH_OTSU):
    if thres == None:
        return df

    values = df.values
    thres_values = []
    thresholding_started = False
    for value in values:
        img = imagify(value, getimage=True)
        th_,img = cv2.threshold(img,cap,255,thres)
        img = [img.flatten()]
        if thresholding_started:
            thres_values = np.concatenate((thres_values, img), axis=0)
        else:
            thres_values = img
            thresholding_started = True

    thres_df = pd.DataFrame(thres_values, columns=df.columns)
    return thres_df


# In[6]:


datacopy = data.copy()
data = apply_thresholding(data, thres=THRESH_BINARY_AND_THRESH_OTSU)
data.head()


# In[7]:


training_features = data.copy()


# In[8]:


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
imputer.fit(training_features)


# In[9]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(training_features)


# In[10]:


from sklearn.decomposition import PCA


# In[11]:


training_features = imputer.transform(training_features)
training_features = scalar.transform(training_features)


# In[12]:


data_labels = load_data(TRAINING_LABELS_FILE)
training_labels = data_labels.values.flatten()


# In[13]:


test_data = load_data(TESTING_FEATURES_FILE)
test_data = apply_thresholding(test_data, thres=THRESH_BINARY_AND_THRESH_OTSU)
testing_features = test_data.copy()
testing_features = imputer.transform(testing_features)
testing_features = scalar.transform(testing_features)


# In[14]:


test_data_labels = load_data(TESTING_LABELS_FILE)
testing_labels = test_data_labels.values.flatten()


# In[15]:


# CNN Classifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

batch_size = 64
epochs = 25
TRAIN_MODEL = True

size = np.int16(np.sqrt(training_features.shape[1]))

train_x = np.reshape(training_features, (-1, size, size, 1))
test_x = np.reshape(testing_features, (-1, size, size, 1))

binarizer = LabelBinarizer()
binarizer.fit(training_labels)
train_y = binarizer.transform(training_labels)
test_y = binarizer.transform(testing_labels)

num_classes = len(binarizer.classes_)
droprate = 0.6

try:
    model = load_model(MODEL_NAME)
except:
    model = None

if model is None:
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(1, 1), strides=(1, 1), activation='elu', input_shape=(size, size, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(4, 4), strides=(1, 1), activation='elu', padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(4, 4), strides=(1, 1), activation='elu', padding='valid'))
    model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(512, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))

    model.add(Dense(256, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))

    model.add(Dense(128, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))

    model.add(Dense(num_classes, activation='softmax'))

else:
    print(MODEL_NAME, " is restored.")

model.summary()

adam = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

callbacks = [EarlyStopping( monitor='val_acc', patience=5, min_delta=0.1, mode='max', verbose=1),
             ModelCheckpoint(MODEL_NAME, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)]

if TRAIN_MODEL:
    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(test_x, test_y),
                        callbacks=callbacks)

    score = model.evaluate(test_x, test_y, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(MODEL_NAME)
else:
    print("Opted not to train the model as TRAIN_MODEL is set to False. May be because model is already trained and is now being used for validation")
    


# In[ ]:


saved_model = load_model(MODEL_NAME)
score = saved_model.evaluate(test_x, test_y, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

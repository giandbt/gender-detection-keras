# import necessary packages
import matplotlib

matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from model.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import json
from keras.models import load_model
import pandas as pd

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (i.e., directory of images)")
args = ap.parse_args()

# initial parameters
batch_size = 64
img_dims = (96, 96, 3)

data = []
labels = []

# get images
csv_path = os.path.join(args.dataset, "wiki_train.csv")
df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    image = cv2.imread(row["Image_Path"])
    if image is None:
        continue
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    data.append(image)
    label = row["Gender"]
    # To be consistent between datasets, woman is 1, men is 0. For WIKI we have to flip
    if label == 1:
        label = 0
    else:
        label = 1
    labels.append(label)


# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20,
                                                  random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# augmenting dataset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

model_path = '/home/giancarlo/Documents/Gender-test/gender-detection-keras/custom.model'
model = load_model(model_path)

# train the model
H = model.evaluate(aug.flow(testX, testY, batch_size=batch_size),verbose=1)
print('Loss: ', H[0])
print('Accuracy: ', H[1])

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
data_dir = os.path.join(args.dataset, 'images')
annotations = os.path.join(args.dataset, 'annotations', 'val2017_400_old.json')
images_files = [os.path.join(data_dir, image) for image in os.listdir(data_dir)]
random.shuffle(images_files)

with open(annotations, 'r') as f:
    annon_dict = json.loads(f.read())

for img in images_files:
    # get images
    img_id, annon, _ = os.path.basename(img)[:-4].split('_')
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    data.append(image)

    # get labels
    label = annon_dict[img_id][annon]['age_gender_pred']['predicted_genders']
    if label == "woman":
        label = 1
    else:
        label = 0
    labels.append(label)

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
trainX = data
trainY = labels

trainY = to_categorical(trainY, num_classes=2)

model_path = '/home/giancarlo/Documents/Gender-test/gender-detection-keras/custom_rigi_2.model'
model = load_model(model_path)

# train the model
H = model.evaluate(x=trainX, y=trainY, batch_size = batch_size,verbose=1)
print('Loss: ', H[0])
print('Accuracy: ', H[1])

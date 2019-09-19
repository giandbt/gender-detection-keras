# import necessary packages
import matplotlib

matplotlib.use("Agg")
from keras.utils import to_categorical
import numpy as np
import argparse
import cv2
import os
from keras.models import load_model
import random

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (i.e., directory of images)")
args = ap.parse_args()

# initial parameters
batch_size = 64
img_dims = (96, 96, 3)

final_images = []
final_labels = []

# get images
woman_paths = os.path.join(args.dataset, "female_names.txt")
man_paths = os.path.join(args.dataset, "male_names.txt")

woman_list = [line.rstrip('\n') for line in open(woman_paths)]
woman_labels = [1]*len(woman_list)
man_list = [line.rstrip('\n') for line in open(man_paths)]
man_labels = [0]*len(man_list)

image_paths = woman_list + man_list
labels = woman_labels + man_labels

data = list(zip(image_paths,labels))
random.shuffle(data)

image_paths = [point[0] for point in data]
labels = [point[1] for point in data]

for idx, image in enumerate(image_paths):
    folder_name = image.split('_')
    if len(folder_name) == 3:
        id_name = folder_name[0] + '_' + folder_name[1]
    elif len(folder_name) == 4:
        id_name = folder_name[0] + '_' + folder_name[1] + '_' + folder_name[2]
    else:
        continue
    image_path = os.path.join(args.dataset, 'lfw', id_name, image)
    img = cv2.imread(image_path)
    if img is None:
        continue
    img = cv2.resize(img, (img_dims[0], img_dims[1]))
    final_images.append(img)
    label = labels[idx]
    final_labels.append(label)


# pre-processing
final_images = np.array(final_images, dtype="float") / 255.0
final_labels = np.array(final_labels)

print(final_labels.shape)

trainX = final_images
trainY = final_labels

trainY = to_categorical(trainY, num_classes=2)

model_path = '/home/giancarlo/Documents/Gender-test/gender-detection-keras/custom_final.model'
model = load_model(model_path)

H = model.evaluate(x=trainX, y=trainY, batch_size = batch_size,verbose=1)
print('Loss: ', H[0])
print('Accuracy: ', H[1])

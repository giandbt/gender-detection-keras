import pandas as pd
import os
import cv2
import json
import shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_custom_csv(data_dir, data_type, output_dir, padding_perc = 0.4):

    images_dir = os.path.join(data_dir, 'images', data_type)
    annotations = os.path.join(data_dir, 'annotations', '%s_400.json' % data_type)

    with open(annotations, 'r') as f:
        annon_dict = json.loads(f.read())

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, 'images'))

    # Initializes variables
    avail_imgs = annon_dict.keys()
    image_paths = []
    age_list = []
    gender_list = []

    # Gets path for all images
    images = [os.path.join(images_dir, image) for image in os.listdir(images_dir) if 'jpg' or 'png' in image]

    for image in images:
        # read image (to determine size later)
        img = cv2.imread(image)

        # gets images Id
        img_id = os.path.basename(image)[:-4].lstrip('0')

        # ensures the image is in the dictionary key
        if not img_id in avail_imgs:
            continue

        for idx, annon in enumerate(annon_dict[img_id].keys()):

            # ensures we have a face detected
            if not annon_dict[img_id][annon]['age_gender_pred']:
                continue

            bbox = annon_dict[img_id][annon]['face']
            x1 = bbox['x1']
            x2 = bbox['x2']
            y1 = bbox['y1']
            y2 = bbox['y2']
            age = annon_dict[img_id][annon]['age_gender_pred']['age']
            gender = annon_dict[img_id][annon]['age_gender_pred']['gender']

            # add padding to face
            upper_y = int(max(0, y1 - (y2 - y1) * padding_perc))
            lower_y = int(min(img.shape[0], y2 + (y2 - y1) * padding_perc))
            left_x = int(max(0, x1 - (x2 - x1) * padding_perc))
            right_x = int(min(img.shape[1], x2 + (x2 - x1) * padding_perc))
            face_im = img[upper_y: lower_y, left_x: right_x, ::-1]

            face_path = os.path.join(output_dir, 'images', '%s_%s_face.jpg' % (img_id, annon))

            Image.fromarray(np.uint8(face_im)).save(os.path.join(face_path))

            image_paths.append(face_path)
            age_list.append(age)
            gender_list.append(gender)

    # saves data in RetinaNet format
    data = {'Image_Path': image_paths, 'Age': age_list, 'Gender': gender_list}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, '%s_labels.csv' % data_type), index=False, header=True)

def display_images(output_dir):
    csv_path = os.path.join(output_dir, 'train2017_labels.csv')
    df = pd.read_csv(str(csv_path))

    for _, row in df.iterrows():
        img_name = row["Image_Path"]
        gender = row["Gender"]
        img = cv2.imread(str(img_name), 1)
        cv2.imshow(gender, img)
        k = cv2.waitKey(0)
        if k == 27:
            break
        cv2.destroyAllWindows()

def filter_face_sizes_combined(data_dir, data_type, area_limit = 400):
    """
    Filters results folder to extract faces of an specific size for training purposes
    :param results_folder:
            str with the folder path for combined results
    :param output_dir:
            str with the folder path we want to save the files
    :param area_limit:
            int regarding maximum size for the face
    :return:
    """

    images_dir = os.path.join(data_dir, 'images', data_type)
    annotations = os.path.join(data_dir, 'annotations', '%s_full.json' % data_type)

    images = [image[:-4] for image in os.listdir(images_dir) if 'jpg' in image]

    with open(annotations, 'r') as file:
        results_dict = json.loads(file.read())

    new_results_dict = dict(results_dict)

    for idx, image in enumerate(images):
        img_id = image.lstrip('0')
        for annon in list(results_dict[img_id].keys()):
            # ensures that there is a face prediction for the annontation
            if not results_dict[img_id][annon]['age_gender_pred']:
                del new_results_dict[img_id][annon]
                if new_results_dict[img_id] == {}:
                    del new_results_dict[img_id]
                continue

            # calculates area
            bbox = results_dict[img_id][annon]['age_gender_pred']['detected']
            area = bbox['height'] * bbox['width']

            # checks the area meets the limit
            if area > area_limit:
                new_results_dict[img_id][annon]['face_area'] = area
            else:
                del new_results_dict[img_id][annon]
                if new_results_dict[img_id] == {}:
                    del new_results_dict[img_id]
    # saves dictionary in a json file
    r = json.dumps(new_results_dict, indent=4)
    with open(os.path.join(data_dir, 'annotations', '%s_%i_test.json' % (data_type, area_limit)), 'w') as f:
        f.write(r)

if __name__ == '__main__':
    data_dir = '/home/giancarlo/Documents/data'
    data_type = 'train2017'
    output_dir = '/home/giancarlo/Documents/Gender-test/custom_final'
    get_custom_csv(data_dir, data_type, output_dir)
    #display_images(output_dir)
    #filter_face_sizes_combined(data_dir, data_type, area_limit=400)
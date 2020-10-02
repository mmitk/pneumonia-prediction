import numpy as np
import cv2
import os

PATH_TO_DIR = r'C:\Users\mmitk\dev\2020\pneumonia\common\data\resampled'

def resample_directory(resampler, dir_path, new_dir_name):

    images, targets = to_numpy_array(dir_path)
    X_res, y_res = resampler.fit_resample(images, targets)
    write_to_directory(new_dir_name, X_res, y_res)


def to_numpy_array(directory_path):
    norm = r'\NORMAL'
    pneum = r'\PNEUMONIA'

    list_images = list()
    list_targets = list()

    norm_path = os.path.join(directory_path, norm)
    pneum_path = os.path.join(directory_path, pneum)

    for image in os.listdir(norm_path):
        image = os.path.join(norm_path, image)
        tmp_im = cv2.imread(image)
        list_images.append(tmp_im)
        list_targets.append(0)

    for image in os.listdir(pneum_path):
        image = os.path.join(pneum_path, image)
        tmp_im = cv2.imread(image)
        list_images.append(tmp_im)
        list_targets.append(1)


    return np.asarray(list_images), np.asarray(list_targets)


def write_to_directory(dir_name, images, targets):
    path = os.path.join(PATH_TO_DIR, dir_name)
    os.makedir(path)

    for count, image in enumerate(images):
        if targets[count] == 0:
            sub_dir = r'\NORMAL'
        else:
            sub_dir = r'PNEUMONIA'
        wpath = os.path.join(path, sub_dir)
        im_name = 'im{}.jpeg'.format(count)
        im =  os.path.join(wpath, im_name)
        cv2.imwrite(im, image)
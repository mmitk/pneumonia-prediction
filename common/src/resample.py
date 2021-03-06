import numpy as np
import cv2
import os
import sys
import shutil
import threading
import time
sys.path.append("C:/Users/mmitk/dev/2020/pneumonia/common/src")
from util import log

PATH_TO_DIR = r'C:\Users\mmitk\dev\2020\pneumonia\common\data\resampled'

def resample_directory(resampler, dir_path, new_dir_name, val = False):

    #/*+ TEMP UNTIL BETTER SOLUTION
    res_dir = r'C:/Users/mmitk/dev/2020/pneumonia/common/data/resampled/{}'.format(new_dir_name)
    if os.path.isdir(res_dir) and len(os.listdir(res_dir)) != 0:
        return
    #/*+start_total = time.time()
    
    print(('-'*10)+'[LOADING IMAGES]'+('-'*10))
    start = time.time()
    images_arr, targets = to_numpy_array(dir_path)
    images_arr = images_arr.reshape(images_arr.shape[0],images_arr.shape[1]*images_arr.shape[2]*images_arr.shape[3])
    targets1 = targets.reshape(-1,1)
    end = time.time()
    #/*+print('\nFinished [LOADING IMAGES]----->Execution Time: {}'.format(str(end - start)))
    print('\nFinished [LOADING IMAGES]')

    start = time.time()
    #/*+load_printer = threading.Thread(target=print_loading,args=())
    print(('-'*10)+'[RESAMPLING IMAGES]'+('-'*10))
    #/*+load_printer.start()
    X_res, y_res = resampler.fit_resample(images_arr, targets1)
    #/*+X_res = X_res.reshape(1,-1)
    #/*+load_printer.join()

    X_res = X_res.reshape(X_res.shape[0],64,64,3)
    y_res = y_res.reshape(1, -1)
    #/*+end = time.time()
    #/*+print('\nFinished [RESAMPLING IMAGES]----->nExecution Time: {}'.format(str(end - start)))
    print('\nFinished [RESAMPLING IMAGES]')

    #start = time.time()
    print(('-'*10)+'[WRITING RESAMPLED IMAGES]'+('-'*10))
    write_to_directory(new_dir_name, X_res, y_res, val)
    log(str(resampler), 'resampled {} to {}'.format(dir_path, new_dir_name), 'MEDIUM')
    #end = time.time()
    #/*+print('\nFinished [WRITING RESAMPLED IMAGES]----->Execution Time: {}'.format(str(end - start)))

    end_total = time.time()

    print('\nFinished [RESAMPLING DIRECTORY]')


def to_numpy_array(directory_path):
    norm = r'NORMAL'
    pneum = r'PNEUMONIA'

    list_images = list()
    list_targets = list()

    norm_path = os.path.join(directory_path, norm)
    pneum_path = os.path.join(directory_path, pneum)

    i = 0

    for image in os.listdir(norm_path):
        if i%1000 == 0 and i != 0:
            print('\n',end='')
        if i%10 == 0:
            print('.',end='')
        image = os.path.join(norm_path, image)
        tmp_im = cv2.imread(image)
        #/*+list_images.append(tmp_im)
        #list_images.append(np.asarray(tmp_im,dtype=int))
        tmp_im = cv2.resize(tmp_im, (64, 64), interpolation=cv2.INTER_CUBIC)
        list_images.append(tmp_im)
        list_targets.append(0)
        i+=1

    for image in os.listdir(pneum_path):
        if i%1000 == 0 and i != 0:
            print('\n',end='')
        if i%10 == 0:
            print('.',end='')
        image = os.path.join(pneum_path, image)
        tmp_im = cv2.imread(image)
        tmp_im = cv2.resize(tmp_im,(64,64), interpolation=cv2.INTER_CUBIC)
        list_images.append(tmp_im)
        #list_images.append(np.asarray(tmp_im, dtype=int))
        list_targets.append(1)
        i+=1


    #return list_images, np.asarray(list_targets)
    return np.asarray(list_images), np.asarray(list_targets)
    #return np.asaarray(list_images, dtype=np.ndarray), np.asanyarray(list_targets)


def write_to_directory(dir_name, images, targets, val):
    path = os.path.join(PATH_TO_DIR, dir_name)
    try:
        os.mkdir(path)
    except Exception as e:
        log(e, 'In resamply.py, line 50', 'LOW')

    if val:
        train_val_dir = r'val'
    else:
        train_val_dir = r'train'

    t_v_path = os.path.join(path, train_val_dir)
    try:
        os.mkdir(t_v_path)
    except Exception as e:
        log(e, 'In resamply.py, line 61', 'LOW')

    i = 0
 
    for count, img in enumerate(images):
        if i%1000 == 0 and i != 0:
            print('\n',end='')

        if i%10 == 0:
            print('.',end='')

        if targets[0][count] == 0:  # *+
            sub_dir = r'NORMAL'
        else:
            sub_dir = r'PNEUMONIA'
        wpath = os.path.join(t_v_path, sub_dir)
        try:
            os.mkdir(wpath)
        except Exception as e:
            log(e, 'In resamply.py, line 75', 'LOW')
        im_name = 'im{}.jpeg'.format(count)
        im_path = r'C:/Users/mmitk/dev/2020/pneumonia/common/data/resampled/{}/{}/{}/{}'.format(dir_name, train_val_dir,
                                                                                                sub_dir, im_name)
        cv2.imwrite(im_path, img)
        i+=1

def remove_resampled_directory(path):
    shutil.rmtree(path)

def print_loading():
    """
    i = 0
    while(True):
        if i == 0:
            print('-',end="\r")
        if i == 1:
            print('/',end="\r")
        if i == 2:
            print('-',end="\r")
        if i == 3:
            print('\\',end="\r")
        i = (i+1)%4
    """
    while True:
        print('-',end="\r")
        print('/',end="\r")
        print('-',end="\r")
        print('\\',end="\r")

    
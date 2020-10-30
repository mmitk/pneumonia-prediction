import numpy as np
import cv2
import os
import sys
import shutil
sys.path.append("C:/Users/mmitk/dev/2020/pneumonia/common/src")
from util import log

PATH_TO_DIR = r'C:\Users\mmitk\dev\2020\pneumonia\common\data\resampled'

class ImageResampler:
    def __init__(self, input_directory = None, target_directory = None, resampler = None):
        self.input_dir = input_directory
        if target_directory is not None:
            self.target_dir = r'C:\Users\mmitk\dev\2020\pneumonia\common\data\resampled\{}'.format(target_directory)
        else:
            self.target_dir = None
        self.res = resampler
        self.orig_shape = None
    
    def resample_directory(self, ImageGenerator=None):
        if os.path.isdir(self.target_dir) and len(os.listdir(self.target_dir)) != 0:
            return


        images_arr, targets = self.generateImageData(ImageGenerator)
        #images1 = images_arr.reshape(-1,1)
        #targets1 = targets.reshape(-1,1)
        #images_arr = images_arr.reshape(-1, 1)
        targets = np.asarray(targets).reshape(-1,1)
        X_res, y_res =self.res.fit_resample(images_arr, targets)
        #X_res = X_res.reshape(1,-1)
        #y_res = y_res.reshape(1, -1)
        try:
            os.mkdir(self.target_dir)
        except Exception as e:
            print(e)
        self.write_to_directory( X_res, y_res)
        log(str(self.res), 'resampled {} to {}'.format(self.input_dir, self.target_dir), 'MEDIUM')

    def generateImageData(self, ImageGenerator):
        data_list = list()
        modified_list = list()
        target_list = list()
        batch_index = 0

        while batch_index <= ImageGenerator.batch_index:
            data = ImageGenerator.next()
            data_list.append(data[0])
            target_list.append(data[1])
            batch_index += 1

        for batch in data_list:
            for img in batch:
                self.orig_shape = img.shape
                #img = np.reshape(img, (img.shape[0], img.shape[2]))
                #img2d = img.transpose(2,0,1).reshape(-1,img.shape[1])
                #test = img2d.reshape(np.roll(img2d.shape,1)).transpose(1,2,0)
                img2d = img.reshape((img.shape[0]*img.shape[1]*img.shape[2]))
                test = img2d.reshape(self.orig_shape)
                modified_list.append(img2d)

        #data_list = np.asarray(data_list)
        targets = [np.argmax(t, axis=1) for t in target_list]
        return modified_list, targets
        


    def write_to_directory(self, images, targets):
        #path = os.path.join(PATH_TO_DIR, self.target_dir)
        try:
            os.mkdir(path)
        except Exception as e:
            log(e, 'In resamply.py, line 50', 'LOW')

  
        try:
            os.mkdir(path)
        except Exception as e:
            log(e, 'In resamply.py, line 61', 'LOW')

        #images = np.reshape(self.orig_shape)
        #targets = np.reshape(self.orig_shape)

        """
        for img in images:
            if img is None:
                continue
            for count, im in enumerate(img):
                if targets[count] == 0: #*+
                    sub_dir = r'NORMAL'
                else:
                    sub_dir = r'PNEUMONIA'
                #wpath = os.path.join(path, sub_dir)
                try:
                    os.mkdir(wpath)
                except Exception as e:
                    log(e, 'In resamply.py, line 75', 'LOW')
                im = np.reshape(im, self.orig_shape)
                im_name = 'im{}.jpeg'.format(count)
                im_path = os.path.join(self.target_dir, sub_dir)
                cv2.imwrite(im_path, im)
        
        """
        targets = targets.tolist()
        for count, img in enumerate(images):
            if targets[count] == 0:  # *+
                sub_dir = r'NORMAL'
            else:
                sub_dir = r'PNEUMONIA'

            img = np.asarray(img).reshape(self.orig_shape)
            img_name = 'img{}.jpeg'.format(count)
            img_path = os.path.join(self.target_dir, sub_dir)
            try:
                os.mkdir(img_path)
            except Exception as e:
                log(e, 'In resamply.py, line 75', 'LOW')
            img_path = os.path.join(img_path, img_name)

            cv2.imwrite(img_path, img)
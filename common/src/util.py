from keras.preprocessing.image import ImageDataGenerator, load_img
import keras
import tensorflow as tf
import os
import time
from pathlib import Path
import datetime

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent # point to common folder
LOGS_DIR = Path(ROOT_DIR / "logs")

def create_train_datagen():
    return ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

def create_test_datagen():
    return ImageDataGenerator(rescale = 1./255)

def create_generator_set(datagen, path, batch_size = 32, shuffle = True):
    #return datagen.flow_from_directory(path, target_size = (64, 64), batch_size = 32, class_mode = 'binary')
    return datagen.flow_from_directory(path, target_size = (64, 64), batch_size = 32, class_mode = 'categorical', shuffle = shuffle)



def log(event, msg, priority):
    filename = str(datetime.date.today()) + '.log'
    path = Path(LOGS_DIR / filename)
    output = str(time.strftime("%H:%M:%S:")) + '{}[{}] -> {}\n'.format(str(event), priority, msg)
    try:
        with open(path, 'a') as f:
            f.write(output)
    except Exception:
        with open(path, 'w+') as f:
            f.write(output)


class ModelException(Exception):
    pass

# Balanced Accuracy Utility Class
class BalancedSparseCategoricalAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)


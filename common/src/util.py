from keras.preprocessing.image import ImageDataGenerator, load_img
import keras
import tensorflow as tf

def create_train_datagen():
    return ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

def create_test_datagen():
    return ImageDataGenerator(rescale = 1./255)

def create_generator_set(datagen, path):
    return datagen.flow_from_directory(path, target_size = (64, 64), batch_size = 32, class_mode = 'binary')

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
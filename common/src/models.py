import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.metrics import BinaryAccuracy
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import classification_report, confusion_matrix
import sklearn
from . import util


class CNNModel:

        def __init__(self):
            self.model = None
            self.curr_accuracy = None

        def summary(self):
            self.model.summary

        def create_model(self):            
            cnn = Sequential()

            #Convolution
            cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

            #Pooling
            cnn.add(MaxPooling2D(pool_size = (2, 2)))

            # 2nd Convolution
            cnn.add(Conv2D(32, (3, 3), activation="relu"))

            # 2nd Pooling layer
            cnn.add(MaxPooling2D(pool_size = (2, 2)))

            # Flatten the layer
            cnn.add(Flatten())

            # Fully Connected Layers
            cnn.add(Dense(activation = 'relu', units = 128))
            cnn.add(Dense(activation = 'sigmoid', units = 1))

            # Compile the Neural network
            cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            self.model = cnn


        def fit_generator(self, generator, validation_generator, epochs=20):
            return self.model.fit_generator(generator, steps_per_epoch = 163, epochs = epochs, validation_data = validation_generator, validation_steps = 624)
        
        def evaluate_model(self, test_generator, test_directory, test_set = None):
            if test_set == None:
                test_set = test_generator.flow_from_directory(test_directory, target_size = (64, 64), batch_size = 32, class_mode = 'binary')
            self.curr_accuracy = self.model.evaluate_generator(test_set, steps = 624)
            return self.curr_accuracy

        def predict_generator(self, test_generator, test_directory, test_set=None):
            if test_set == None:
                test_set = test_generator.flow_from_directory(test_directory, target_size = (64, 64), batch_size = 32, class_mode = 'binary')
            return self.model.predict_generator(test_set, 100)
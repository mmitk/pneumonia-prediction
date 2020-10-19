import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.metrics import BinaryAccuracy
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import numpy as np
from . import util

METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]

class CNNModel:        
        def __init__(self, early_stopping = False):
            self.early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0.1, patience=10)
            self.model = None
            self.curr_accuracy = None
            self.history = None
            self.early_stopping = early_stopping

        def summary(self):
            self.model.summary

        def create_model(self, metrics = METRICS):            
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
            #/*+cnn.add(Dense(activation = 'sigmoid', units = 1))
            cnn.add(Dense(activation = 'sigmoid', units = 2))

            # Compile the Neural network
            cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = metrics)
            #/*+cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            self.model = cnn


        def fit_generator(self, generator, validation_generator, epochs=50, steps_per_epoch=163):
            if self.early_stopping:
                self.history = self.model.fit_generator(generator, steps_per_epoch = steps_per_epoch, epochs = epochs, validation_data = validation_generator, validation_steps = 624, callbacks=[self.early_stop])
            else:
                self.history = self.model.fit_generator(generator, steps_per_epoch = steps_per_epoch, epochs = epochs, validation_data = validation_generator, validation_steps = 624)
            #/*+self.history = self.model.fit_generator(generator, steps_per_epoch = 624 // 32, epochs = epochs, validation_data = validation_generator, validation_steps = 624 // 32, callbacks=[self.early_stop])
            return self.history
            
        def evaluate_model(self, test_generator=None, test_directory=None, test_set = None):
            if test_set == None:
                test_set = test_generator.flow_from_directory(test_directory, target_size = (64, 64), batch_size = 32, class_mode = 'categorical')
            self.curr_accuracy = self.model.evaluate_generator(test_set, steps = 624)
            return self.curr_accuracy

        def predict_generator(self, test_generator=None, test_directory=None, test_set=None):
            #if test_set == None:
                #test_set = test_generator.flow_from_directory(test_directory, target_size = (64, 64), batch_size = 32, class_mode = 'binary')
            test_set = test_generator.flow_from_directory(test_directory, target_size = (64, 64), batch_size = 32, class_mode = 'categorical')
 
            filenames = test_set.filenames
            nb_samples = len(filenames)
            #/*+return self.model.predict_generator(test_set, steps = nb_samples)
            #/*+return self.model.predict_generator(test_set, 100)
            return self.model.predict_generator(test_set, 624 // 33)

        def plot_history(self):
            if self.history is None:
                raise util.ModelException('History is None')
            history = self.history
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            metrics =  ['loss', 'auc', 'precision', 'recall']
            for n, metric in enumerate(metrics):
                name = metric.replace("_"," ").capitalize()
                plt.subplot(2,2,n+1)
                plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
                plt.plot(history.epoch, history.history['val_'+metric],
                color=colors[0], linestyle="--", label='Val')
                plt.xlabel('Epoch')
                plt.ylabel(name)
                if metric == 'loss':
                    plt.ylim([0, plt.ylim()[1]])
                elif metric == 'auc':
                    plt.ylim([0.8,1])
                else:
                    plt.ylim([0,1])

                plt.legend()

        def display_confusion_matrix(self,test_data_generator):
            """
            Y_pred = self.predict_generator(test_set=test_set)
            y_pred = np.argmax(Y_pred, axis=1)
            #print('Y_pred: {}\n '.format(Y_pred.shape))
            #print('test_set:{}'.format(test_set.classes.shape))
            #print('test_set:{}'.format(test_set.classes.shape))
            """
            

            test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
            predictions = self.model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
            y_pred = np.argmax(predictions, axis=1)
            true_classes = test_data_generator.classes
            class_labels = list(test_data_generator.class_indices.keys())
            #print(sklearn.metrics.classification_report(true_classes, y_pred, target_names=class_labels))

            b_score = balanced_accuracy_score(true_classes, y_pred)

            cm = confusion_matrix(test_data_generator.classes, y_pred)
            #/*+cm = confusion_matrix(validation_generator.classes, y_pred)

            plt.figure(figsize=(9,9))
            sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
            plt.ylabel('Actual label');
            plt.xlabel('Predicted label');
            all_sample_title = 'Balanced Accuracy Score: {0}'.format(b_score)
            plt.title(all_sample_title, size = 15);
            
        def get_classification_report(self, test_data_generator):
            test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)
            predictions = self.model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
            y_pred = np.argmax(predictions, axis=1)
            true_classes = test_data_generator.classes
            class_labels = list(test_data_generator.class_indices.keys())
            print(sklearn.metrics.classification_report(true_classes, y_pred, target_names=class_labels))
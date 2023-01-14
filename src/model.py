import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  
from keras.callbacks import EarlyStopping
import configparser
import numpy as np

class breed_prediction():
    def __init__(self):
        super(breed_prediction,self).__init__()
        self.config = configparser.ConfigParser()
        # read config to fetch different parameters and paths
        self.config.read("../config.ini")


    def build_model(self,train_tensors):
        self.model = Sequential()
        self.model.add(Conv2D(input_shape=train_tensors.shape[1:],filters=16,kernel_size=2, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=32,kernel_size=2, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(tf.keras.layers.Dropout(0.1))
        self.model.add(Conv2D(filters=32,kernel_size=2, activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(tf.keras.layers.Dropout(0.1))
        self.model.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
        self.model.add(MaxPooling2D())

        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(int(self.config["global"]["class_labels"]),activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

    def train(self,train_tensors, train_Y,valid_tensors, valid_Y): 
        earlyStopping = EarlyStopping(monitor='val_accuracy',restore_best_weights=True,
                                    patience=int(self.config["global"]["patience"]), verbose=0, mode='max'  ,min_delta=0.001)
        checkpointer = ModelCheckpoint(filepath=self.config["global"]["save_model_own"], verbose=1, save_best_only=True)

        self.model.fit(train_tensors, train_Y, 
                  validation_data=(valid_tensors, valid_Y),
                  epochs=int(self.config["global"]["epochs"]), batch_size=int(self.config["global"]["batch_size"]), callbacks=[checkpointer,earlyStopping], verbose=1)


    def test(self,test_tensors,test_Y):
        dog_breed_predictions = [np.argmax(self.model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

        # test accuracy of the model
        test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_Y, axis=1))/len(dog_breed_predictions)
        return test_accuracy
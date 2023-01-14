import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  
from keras.callbacks import EarlyStopping
import configparser
import argparse
import numpy as np
import random as rn

rn.seed(0)
np.random.seed(0)

class breed_prediction_trasferlearning():
    def __init__(self):
        super(breed_prediction_trasferlearning,self).__init__()
        self.config = configparser.ConfigParser()
        # read config to know path to store log file
        self.config.read("../config.ini")

        bottleneck_features = np.load(self.config["global"]["bottleneck_features"])
        self.train_Xception = bottleneck_features['train']
        self.valid_Xception = bottleneck_features['valid']
        self.test_Xception = bottleneck_features['test']

    def build_model(self):
        self.Xception_model = Sequential()
        self.Xception_model.add(GlobalAveragePooling2D(input_shape=self.train_Xception.shape[1:]))
        self.Xception_model.add(tf.keras.layers.Dropout(0.1))
        self.Xception_model.add(tf.keras.layers.BatchNormalization())
        self.Xception_model.add(Dense(int(self.config["global"]["class_labels"]), activation='softmax'))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.Xception_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(self.Xception_model.summary())
    
    
    
   
    def train(self,train_Y,valid_Y):
        checkpointer = ModelCheckpoint(filepath=self.config["global"]["save_model_transfer"], 
                                       verbose=1, save_best_only=True)

        earlyStopping = EarlyStopping(monitor='val_accuracy',
                                                   restore_best_weights=True,
                                                   patience=int(self.config["global"]["patience"]), verbose=0, mode='max'  ,min_delta=0.01
                                                   )

        self.Xception_model.fit(self.train_Xception, train_Y, 
                  validation_data=(self.valid_Xception, valid_Y),
                  epochs=int(self.config["global"]["epochs"]), batch_size=int(self.config["global"]["batch_size"]), callbacks=[checkpointer,earlyStopping], verbose=1)
    
    def test(self,test_Y):
        Xception_predictions = [np.argmax(self.Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in self.test_Xception]

        # test accuracy of the Xception model Transfer Learning
        test_accuracy = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_Y, axis=1))/len(Xception_predictions)
        return test_accuracy

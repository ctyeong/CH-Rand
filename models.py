import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np


class Cls(keras.Model):
    def __init__(self, n_cls=1):
        super(Cls, self).__init__()
    
        l2_norm = .001 
        act_func = layers.LeakyReLU(alpha=.1) 

        self.conv1 = layers.Conv2D(64, 3, padding='same', activation=act_func, 
                        kernel_regularizer=keras.regularizers.l2(l2_norm))
        self.norm1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2, padding='same')

        self.conv2 = layers.Conv2D(128, 3, padding='same', activation=act_func, 
                        kernel_regularizer=keras.regularizers.l2(l2_norm))
        self.norm2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2, padding='same')

        self.conv3 = layers.Conv2D(256, 3, padding='same', activation=act_func, 
                        kernel_regularizer=keras.regularizers.l2(l2_norm))
        self.norm3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(2, padding='same')

        self.conv4 = layers.Conv2D(512, 3, padding='same', activation=act_func,
                        kernel_regularizer=keras.regularizers.l2(l2_norm))
        self.norm4 = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D(2, padding='same')

        self.conv5 = layers.Conv2D(512, 3, padding='same', activation=act_func, 
                        kernel_regularizer=keras.regularizers.l2(l2_norm))
        self.norm5 = layers.BatchNormalization()
        self.pool5 = layers.AveragePooling2D(2, padding='same')

        self.flatten = layers.Flatten()
        self.fc0 = layers.Dense(256, activation=act_func)
        
        if n_cls == 1:
            self.fc = layers.Dense(n_cls, activation='sigmoid', dtype='float32')
            
        else:
            self.fc = layers.Dense(n_cls, activation='softmax', dtype='float32')


    def feature(self, inputs, training=False, fc_out=False):
        x = self.conv1(inputs, training=training)
        x = self.norm1(x, training=training)
        x = self.pool1(x)
        x = self.conv2(x, training=training)
        x = self.norm2(x, training=training)
        x = self.pool2(x)
        x = self.conv3(x, training=training)
        x = self.norm3(x, training=training)
        x = self.pool3(x)
        x = self.conv4(x, training=training)
        x = self.norm4(x, training=training)
        x = self.pool4(x)
        x = self.conv5(x, training=training)
        x = self.norm5(x, training=training)
        x = self.pool5(x)
        x = self.flatten(x)
        
        if fc_out:
            x = self.fc0(x)
        
        return x
    
    
    def call(self, inputs, training=None):
        x = self.feature(inputs, training=training)
        x = self.fc0(x)
        output = self.fc(x)
    
        return output

    
class LeNet(keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu')
        self.pool1 = layers.AveragePooling2D()
        self.conv2 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')
        self.pool2 = layers.AveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units=120, activation='relu')
        self.fc2 = layers.Dense(units=84, activation='relu')
        self.fc3 = layers.Dense(units=1, activation='sigmoid', dtype='float32')

        
    def feature(self, inputs, training=False):
        
        x = self.conv1(inputs, training=training)
        x = self.pool1(x)
        x = self.conv2(x, training=training)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        
        return x

    
    def call(self, inputs, training=None, mask=None):
        x = self.feature(inputs, training)
        output = self.fc3(x)
    
        return output

        
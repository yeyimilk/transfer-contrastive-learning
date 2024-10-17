
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Activation, Add, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow import keras

from .base_models import NN
from abc import abstractmethod

def build_classifier_model(encoder, out):
    model = tf.keras.Sequential([
            encoder, 
            Dense(out, activation='softmax')
    ])
    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model, encoder


class BaseNN(NN):
    def __init__(self, filters, kernels, dilations, dense, shape, name, out):
        self.filters = filters
        self.kernels = kernels
        self.dilations = dilations
        self.dense = dense
        self.shape = shape
        self.name = name
        self.out = out
        
        super().__init__(shape)
    
    @abstractmethod
    def build_encoder(self, x):
        pass
    
    def build_model(self):
        input_tensor = Input(shape=self.shape)
        if 'mlp' not in self.name:
            input_tensor = Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_tensor)
        encoder = self.build_encoder(input_tensor)
        model, encoder = build_classifier_model(encoder, self.out)
        self.encoder = encoder
        self.model = model
        
    def load_weights_for_encoder(self, path):
        print(f"Loading weights from {path}")
        self.encoder.load_weights(path)

class StraightNN(BaseNN):
    def __init__(self, filters, kernels, dilations, dense, shape, out=3, randomize=False):
        self.randomize = randomize
        super().__init__(filters, kernels, dilations, dense, shape, f"straight_nn_{'_'.join([str(l) for l in filters])}", out)
    
    def build_encoder(self, input_tensor):
        x = input_tensor
        for i in range(len(self.filters)):
            if self.randomize:
                x = Conv1D(filters=self.filters[i], kernel_size=self.kernels[i],
                           dilation_rate=self.dilations[i],
                           padding='causal', 
                           activation='relu', 
                           kernel_initializer=tf.keras.initializers.RandomNormal())(x)
            else:
                x = Conv1D(filters=self.filters[i], kernel_size=self.kernels[i],
                        dilation_rate=self.dilations[i],
                        padding='causal', 
                        activation='relu')(x)
            
        
        x = Lambda(lambda x: tf.reduce_max(x, axis=-1))(x)
        if self.randomize:
            x = Dense(self.dense, activation='relu',
                      kernel_initializer=tf.keras.initializers.RandomNormal())(x)
        else:
            x = Dense(self.dense, activation='relu')(x)
        
        return Model(inputs=input_tensor, outputs=x)

def residual_block(x, filters, kernel_size, dilation_rate):
    # Shortcut connection
    shortcut = x
    
    # Main path
    x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(x)
    x = Activation('relu')(x)
    
    # Adjusting the shortcut channel dimension if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(shortcut)
    
    # Adding the shortcut to the output
    x = Add()([shortcut, x])
    
    return x

class ResNN(BaseNN):
    def __init__(self, filters, kernels, dilations, dense, shape, out=3):
        super().__init__(filters, kernels, dilations, dense, shape, f"res_nn_{'_'.join([str(l) for l in filters])}_{dense}", out)
        
    def build_encoder(self, input_tensor):
        x = input_tensor
        for i in range(len(self.filters)):
            x = residual_block(x, self.filters[i], self.kernels[i], self.dilations[i])
        
        x = tf.keras.layers.Flatten()(x)
        
        if self.dense > 0:
            x = Dense(self.dense, activation='relu')(x)
        
        return Model(inputs=input_tensor, outputs=x)


class MLP_(BaseNN):
    def __init__(self, layers, shape, out=3):
        self.layers = layers
        self.name = f"mlp_{'_'.join([str(l) for l in layers])}"
        super().__init__([], [], [], [], shape, self.name, out)
    
    def build_encoder(self, input_tensor):
        x = input_tensor
        for i in range(len(self.layers)):
            x = keras.layers.Dense(self.layers[i], activation=tf.nn.relu)(x)
        
        return Model(inputs=input_tensor, outputs=x)


class RNN_(BaseNN):
    def __init__(self, layers, shape, return_sequences, flat, r_unit, out):
        self.layers = layers
        self.return_seq = return_sequences
        self.flat = flat
        
        if r_unit == 'SimpleRNN':
            self.R_Unit = keras.layers.SimpleRNN
        elif r_unit == 'LSTM':
            self.R_Unit = keras.layers.LSTM
        else:
            self.R_Unit = keras.layers.GRU
        
        self.name = f"rnn_{'_'.join([str(l) for l in layers])}_{'_'.join([str(l) for l in return_sequences])}_flat_{flat}_r_unit_{r_unit}"
        super().__init__([], [], [], [], shape, self.name, out)
    
    def build_encoder(self, input_tensor): 
        x = input_tensor
        for i in range(len(self.return_seq)):
            x = self.R_Unit(self.layers[i], return_sequences=self.return_seq[i])(x)
            
        if self.flat:
            x = keras.layers.Flatten()(x)
        else:
            x = keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1))(x)
        
        for num in range(len(self.return_seq), len(self.layers)):
            x = keras.layers.Dense(num, activation=tf.nn.relu)(x)
        
        return Model(inputs=input_tensor, outputs=x)



class CNN_O(BaseNN):
    def __init__(self,  shape, out=3):
        super().__init__([], [], [], [], shape, 'cnn_o', out)
    
    def build_encoder(self, input_tensor):
        x = input_tensor
        x = keras.layers.Conv1D(filters=10, kernel_size=10, strides=2, activation='relu')(x)
        x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
        x = keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(100, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        return Model(inputs=input_tensor, outputs=x)
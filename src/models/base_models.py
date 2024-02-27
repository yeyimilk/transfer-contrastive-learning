import tensorflow as tf
from tensorflow import keras
from abc import ABC, abstractmethod

class NN(ABC):
    def __init__(self, shape):
        self.shape = shape
        self.build_model()
        self.compile_model()

    def compile_model(self):
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        self.model.build(self.shape)

    @abstractmethod
    def build_model(self):
        pass

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=0,
            validation_data=None,
            validation_batch_size=None,
            validation_freq=100):
        return self.model.fit(x=x,y=y,batch_size=batch_size,epochs=epochs,verbose=verbose,validation_data=validation_data,validation_batch_size=validation_batch_size,validation_freq=validation_freq)

    def predict(self, x_test, verbose=1):
        return self.model.predict(x_test, verbose=verbose)

    def score(self, x_test, y_test, verbose=1):
        # Evaluate the model
        result = self.model.evaluate(x_test, y_test, verbose=verbose)
        return result[1]


class CNNModel(NN):
    def __init__(self, case, shape):
        self.case = case
        self.name = f"cnn_{case}"
        super().__init__(shape)
        
    def build_case0(self):
        # https://www.sciencedirect.com/science/article/pii/S1386142521003085
        # Classifying breast cancer tissue by Raman spectroscopy with one-dimensional convolutional neural network
        # Define the model
        model = keras.models.Sequential([
            keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
            keras.layers.Conv1D(filters=10, kernel_size=10, strides=2, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2, strides=2),
            keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(3, activation='softmax')
        ])
        self.model = model
    
    def build_case1(self):
        model = keras.models.Sequential([
            keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
            keras.layers.Conv1D(filters=500, kernel_size=5, activation='relu'),
            keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1)),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])
        self.model = model

    def build_model(self):
        if self.case == 0:
            self.build_case0()
        else:
            self.build_case1()

class MLP(NN):
    def __init__(self, layers, shape):
        self.layers = layers
        self.name = f"mlp_{'_'.join([str(l) for l in layers])}"
        super().__init__(shape=shape)
        

    def build_model(self):
        sequences = [keras.layers.Dense(l, activation=tf.nn.relu) for l in self.layers]
        sequences.append(keras.layers.Dense(3, activation='softmax'))
        model = keras.Sequential(sequences)
        self.model = model


class LSTM(NN):
    def __init__(self, layers, shape):
        self.layers = layers
        self.name = f"lstm_{'_'.join([str(l) for l in layers])}"
        super().__init__(shape)
        

    def build_model(self):
        sequences = [
            keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1)),
            keras.layers.LSTM(self.layers[0], return_sequences=True),
            keras.layers.Flatten()
        ]
        
        for num in range(1, len(self.layers)):
            sequences.append(keras.layers.Dense(num, activation=tf.nn.relu))

        sequences.append(keras.layers.Dense(3, activation='softmax'))
        model = keras.Sequential(sequences)
        self.model = model

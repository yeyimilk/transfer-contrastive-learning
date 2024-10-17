
import tensorflow as tf
from tensorflow import keras
import random

class TimeShift(tf.keras.layers.Layer):
    def __init__(self, shift, enable_random=False, **kwargs):
        super(TimeShift, self).__init__(**kwargs)
        self.shift = abs(shift)
        self.enable_random = enable_random
        

    def call(self, inputs):
        if self.enable_random and random.random() < 0.5:
            return inputs
        
        shift = random.randint(-self.shift, self.shift)
        return tf.roll(inputs, shift=shift, axis=1)


class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, factor, enable_random=False, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.factor = factor
        self.enable_random = enable_random

    def call(self, inputs):
        if self.enable_random and random.random() < 0.2:
            return inputs
        
        factor = random.uniform(1-self.factor, 1+self.factor)
        return inputs * factor


class GaussianNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, std, mean=0, enable_random=False, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.std = std
        self.mean = mean
        self.enable_random = enable_random

    def call(self, inputs):
        if self.enable_random and random.random() < 0.2:
            return inputs
        
        noise = tf.random.normal(shape=tf.shape(inputs), mean=self.mean, stddev=self.std)
        noisy_inputs = inputs + noise
        
        min_value = tf.reduce_min(noisy_inputs)
        # Add a small epsilon for numerical stability
        epsilon = 1e-7
        
        if min_value < 0:
            noisy_inputs_positive = noisy_inputs + tf.abs(min_value) + epsilon
        else:
            noisy_inputs_positive = noisy_inputs
        return noisy_inputs_positive


class CoefficientsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CoefficientsLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.coefficients = self.add_weight(shape=input_shape[1:], initializer='random_uniform', trainable=True)

    def call(self, inputs):
        return inputs * self.coefficients


class ReverseLayer(tf.keras.layers.Layer):
    def __init__(self, length=None, enable_random=False, **kwargs):
        super(ReverseLayer, self).__init__(**kwargs)
        self.length = length
        self.enable_random = enable_random
    
    def call(self, inputs):
        if self.length is None:
            return inputs
        
        if self.enable_random and random.random() < 0.3:
            return inputs
    
        length = random.randint(0, self.length)
        shape_size = len(inputs.shape)
        if shape_size == 2:
            start = random.randint(0, inputs.shape[1] - length)
            if random.random() < 0.5:
                start = 0
                length = inputs.shape[1]
            end = start + length
            
            # Slice out the part of each row that you want to reverse
            sliced_inputs = inputs[:, start:end]

            # Reverse the sliced inputs along the column axis
            reversed_slice = tf.reverse(sliced_inputs, axis=[1])

            # Concatenate the inputs before the start, the reversed slice, and the inputs after the end
            outputs = tf.concat([inputs[:, :start], reversed_slice, inputs[:, end:]], axis=1)
            return outputs
     

def get_layer_augmenter(shape, augs):
    (shift, scale, noise, reverse) = augs
    return keras.Sequential([
        keras.Input(shape=shape),
        TimeShift(shift),
        ScaleLayer(scale),
        GaussianNoiseLayer(noise),
        ReverseLayer(reverse)
    ])
    
def get_random_layer_augmenter(shape, augs):
    (shift, scale, noise, reverse) = augs
    return keras.Sequential([
        keras.Input(shape=shape),
        TimeShift(shift, enable_random=True),
        ScaleLayer(scale, enable_random=True),
        GaussianNoiseLayer(noise, enable_random=True),
        ReverseLayer(reverse, enable_random=True)
    ])
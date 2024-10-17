import tensorflow as tf
from tensorflow import keras

def get_linear_probe(mlps, activation=tf.nn.softmax):
    sequences = []
    for i in range(len(mlps)-1):
        sequences.append(keras.layers.Dense(mlps[i], activation=tf.nn.relu))
    if activation is not None:
        sequences.append(keras.layers.Dense(mlps[len(mlps) - 1], activation=tf.nn.softmax))
    else:
        sequences.append(keras.layers.Dense(mlps[len(mlps) - 1]))
    return keras.Sequential(
                sequences, name="linear_probe"
            )
    
def get_mlp_header(mlps):
    sequences = []
    for i in range(0, len(mlps)-1):
        sequences.append(keras.layers.Dense(mlps[i], activation=tf.nn.relu))
    sequences.append(keras.layers.Dense(mlps[len(mlps) - 1]))
    return keras.Sequential(
                sequences,
                name="projection_head"
            )
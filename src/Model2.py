import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses, metrics, activations
from tensorflow.contrib.layers import batch_norm, dropout
from tensorflow.contrib.opt import NadamOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.saved_model.model_utils.mode_keys import is_train

def RRelu(x):
    rand = tf.Variable(tf.random_uniform([]) * 0.3 + 0.1, dtype=tf.float32)
    alpha = tf.cond(is_train, lambda: rand, lambda: tf.Variable(0.3, dtype=tf.float32))
    return tf.nn.relu(x) - tf.nn.relu(-x)*alpha

def conv_layer(layer, num_filters, k_size=(3, 3), shape=(-1, 28, 28, 1), padding="same"):
    new_layer = Conv2D(num_filters, kernel_size=k_size, strides=(1, 1), padding=padding,
                       kernel_initializer='he_normal', input_shape=shape)(layer)
    new_layer = batch_norm(new_layer, updates_collections=None, center=True, scale=True)
    return activations.relu(new_layer)

def fc_layer(layer, num_neurons):
    new_layer = Dense(num_neurons, kernel_initializer='he_normal')(layer)
    new_layer = batch_norm(new_layer, updates_collections=None, center=True, scale=True)
    return activations.relu(new_layer)

def Fashion_CNN(input_shape, num_classes, learning_rate, graph):

    with graph.as_default():

        #is_train = tf.placeholder(tf.bool)
        img = tf.placeholder(tf.float32, input_shape)

        labels = tf.placeholder(tf.float32, shape=(None, num_classes))
        lr = tf.placeholder(tf.float32)

        # first 3 convolutions approximate Conv(7,7):
        layer = conv_layer(img, 64)
        layer = conv_layer(layer, 64)
        layer = conv_layer(layer, 64)
        layer = MaxPooling2D()(layer)
        layer = dropout(layer, keep_prob=0.7)
        layer = conv_layer(layer, 128, shape=(-1, 14, 14, -1))
        layer = conv_layer(layer, 128, shape=(-1, 14, 14, -1))
        layer = conv_layer(layer, 64, (1, 1), shape=(-1, 14, 14, -1))
        layer = MaxPooling2D()(layer)
        layer = Flatten()(layer)
        layer = dropout(layer, keep_prob=0.7)
        layer = fc_layer(layer, 2048)
        layer = dropout(layer)
        layer = fc_layer(layer, 512)
        layer = dropout(layer)
        layer = fc_layer(layer, 256)
        layer = dropout(layer)
        layer = Dense(10, kernel_initializer='glorot_normal')(layer)
        layer = batch_norm(layer, updates_collections=None, center=True, scale=True)
        preds = activations.softmax(layer)

        lossL2 = tf.add_n(
        [ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name ])

        beta = 1e-7
        loss = tf.reduce_mean(losses.categorical_crossentropy(labels, preds))
        train_step = NadamOptimizer(learning_rate=lr).minimize(loss)

        acc_value = tf.reduce_mean(metrics.categorical_accuracy(labels, preds))

        return img, labels, lr, train_step, loss, acc_value

import tensorflow as tf
import tensorflow.contrib.slim as slim
import json
import numpy as np
from functools import partial
from scipy import stats
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, './utils'))
import tf_util

batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, scope='bn', updates_collections=None)

def leak_relu(x, leak=0.1, scope=None):
    return tf.where(x >= 0, x, leak * x)

def get_encoder_network(images, is_training):
    # Args:
    #     images: is tensor of size BxHxWx1 where B is batch size, H and W are image dimensions
    # Returns:
    #     encoder: is a Python dictionary containing all tensors
    net = {}
    net['conv1'] = tf_util.conv2d(images, 32, [3,3], activation_fn=leak_relu, scope='encoder/conv1', bn=True, is_training=is_training) # H * W
    net['conv2'] = tf_util.conv2d(net['conv1'], 32, [3,3], activation_fn=leak_relu, scope='encoder/conv2', bn=True, is_training=is_training) # H * W
    net['pool3'] = tf_util.max_pool2d(net['conv2'], [2,2], scope='encoder/pool3', stride=[2,2], padding='VALID') # H/2 * W/2
    net['conv4'] = tf_util.conv2d(net['pool3'], 32, [3,3], activation_fn=leak_relu, scope='encoder/conv4', bn=True, is_training=is_training) # H/2 * W/2
    net['conv5'] = tf_util.conv2d(net['conv4'], 32, [3,3], activation_fn=leak_relu, scope='encoder/conv5', bn=True, is_training=is_training) # H/2 * W/2
    net['pool6'] = tf_util.max_pool2d(net['conv5'], [2,2], scope='encoder/pool6', stride=[2,2], padding='VALID') # H/4 * W/4
    net['conv7'] = tf_util.conv2d(net['pool6'], 64, [3,3], activation_fn=leak_relu, scope='encoder/conv7', bn=True, is_training=is_training) # H/4 * W/4
    net['conv8'] = tf_util.conv2d(net['conv7'], 64, [3,3], activation_fn=leak_relu, scope='encoder/conv8', bn=True, is_training=is_training) # H/4 * W/4
    net['conv9'] = tf_util.conv2d(net['conv8'], 64, [3,3], activation_fn=leak_relu, scope='encoder/conv9', bn=True, is_training=is_training) # H/4 * W/4
    net['pool10'] = tf_util.max_pool2d(net['conv9'], [2,2], scope='encoder/pool10', stride=[2,2], padding='VALID') # H/8 * W/8
    net['conv11'] = tf_util.conv2d(net['pool10'], 64, [3,3], activation_fn=leak_relu, scope='encoder/conv11', bn=True, is_training=is_training) # H/8 * W/8
    net['conv12'] = tf_util.conv2d(net['conv11'], 64, [3,3], activation_fn=leak_relu, scope='encoder/conv12', bn=True, is_training=is_training) # H/8 * W/8
    net['conv13'] = tf_util.conv2d(net['conv12'], 64, [3,3], activation_fn=leak_relu, scope='encoder/conv13', bn=True, is_training=is_training) # H/8 * W/8
    net['pool14'] = tf_util.max_pool2d(net['conv13'], [2,2], scope='encoder/pool14', stride=[2,2], padding='VALID') # H/16 * W/16
    net['conv15'] = tf_util.conv2d(net['pool14'], 128, [3,3], activation_fn=leak_relu, scope='encoder/conv15', bn=True, is_training=is_training) # H/16 * W/16
    net['conv16'] = tf_util.conv2d(net['conv15'], 128, [3,3], activation_fn=leak_relu, scope='encoder/conv16', bn=True, is_training=is_training) # H/16 * W/16
    net['conv17'] = tf_util.conv2d(net['conv16'], 128, [3,3], activation_fn=leak_relu, scope='encoder/conv17', bn=True, is_training=is_training) # H/16 * W/16
    net['pool18'] = tf_util.max_pool2d(net['conv17'], [2,2], scope='encoder/pool18', stride=[2,2], padding='VALID') # H/32 * W/32
    net['conv19'] = tf_util.conv2d(net['pool18'], 128, [3,3], activation_fn=leak_relu, scope='encoder/conv19', bn=True, is_training=is_training) # H/32 * W/32

    return net

def get_decoder_network(features, encoder, is_training):
    # Args:
    #     features: is tensor of size BxHxWxF where B is batch size, H, W and F are features dimensions
    #     encoder: the Python dictionary containning all tensors from get_encoder_network
    # Returns:
    #     edge_logits: is tensor of size BxVxHxWx1 where B is batch size, V is number of views, H and W are image dimensions
    net = {}
    net['conv20'] = tf_util.conv2d_transpose(features, 128, [4,4], activation_fn=leak_relu, bn=True, is_training=is_training, scope='decoder/conv20', stride=[2,2], padding='SAME') # H/16 * H/16
    concat = tf.concat(axis=3, values=[encoder['conv17'], net['conv20']])
    net['conv21'] = tf_util.conv2d(concat, 128, [3,3], activation_fn=leak_relu, scope='decoder/conv21', bn=True, is_training=is_training) # H/16 * H/16
    net['conv22'] = tf_util.conv2d_transpose(net['conv21'], 64, [4,4], activation_fn=leak_relu, bn=True, is_training=is_training, scope='decoder/conv22', stride=[2,2], padding='SAME') # H/8 * H/8
    concat = tf.concat(axis=3, values=[encoder['conv13'], net['conv22']])
    net['conv23'] = tf_util.conv2d(concat, 64, [3,3], activation_fn=leak_relu, scope='decoder/conv23', bn=True, is_training=is_training) # H/8 * H/8
    net['conv24'] = tf_util.conv2d_transpose(net['conv23'], 64, [4,4], activation_fn=leak_relu, bn=True, is_training=is_training, scope='decoder/conv24', stride=[2,2], padding='SAME') # H/4 * H/4
    concat = tf.concat(axis=3, values=[encoder['conv9'], net['conv24']])
    net['conv25'] = tf_util.conv2d(concat, 64, [3,3], activation_fn=leak_relu, scope='decoder/conv25', bn=True, is_training=is_training) # H/4 * H/4
    net['conv26'] = tf_util.conv2d_transpose(net['conv25'], 64, [4,4], activation_fn=leak_relu, bn=True, is_training=is_training, scope='decoder/conv26', stride=[2,2], padding='SAME') # H/2 * H/2
    concat = tf.concat(axis=3, values=[encoder['conv5'], net['conv26']])
    net['conv27'] = tf_util.conv2d(concat, 64, [3,3], activation_fn=leak_relu, scope='decoder/conv27', bn=True, is_training=is_training) # H/2 * H/2
    net['conv28'] = tf_util.conv2d_transpose(net['conv27'], 32, [4,4], activation_fn=leak_relu, bn=True, is_training=is_training, scope='decoder/conv28', stride=[2,2], padding='SAME') # H * W
    concat = tf.concat(axis=3, values=[encoder['conv2'], net['conv28']])
    net['conv29'] = tf_util.conv2d(concat, 32, [3,3], activation_fn=leak_relu, scope='decoder/conv29', bn=True, is_training=is_training) # H * W
    edge_logits = tf_util.conv2d(net['conv29'], 1, [3,3], activation_fn=None, scope='decoder/edge_logit', bn=False, is_training=is_training)

    return edge_logits

def get_model(images, is_training):
    # Args:
    #     images is tensor of size BxVxHxWx1 where B is batch size, V is number of views, H and W are image dimensions
    # Returns:
    #     edge_logits: is tensor of size BxVxHxWx1 where B is batch size, V is number of views, H and W are image dimensions

    # Get batch information
    batch_size = images.get_shape()[0].value
    num_views = images.get_shape()[1].value
    image_height = images.get_shape()[2].value
    image_width = images.get_shape()[3].value

    # Reshape input tensor to fit 2D convolution
    images = tf.reshape(images, [batch_size * num_views, image_height, image_width, 1])

    # Encoder
    encoder = get_encoder_network(images, is_training)

    # Flatten the last layer as 1D features
    feature_height = encoder['conv19'].get_shape()[1].value
    feature_width = encoder['conv19'].get_shape()[2].value
    feature_channel = encoder['conv19'].get_shape()[3].value
    num_features = feature_height * feature_width * feature_channel
    features = tf.reshape(encoder['conv19'], [batch_size, num_views, num_features])

    # Recurrent Neural Network
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256, 256]]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    # Forward sequence
    lstm_outputs_forward, _ = tf.nn.dynamic_rnn(multi_rnn_cell, features, dtype=tf.float32, sequence_length=tf.fill([batch_size], num_views))
    # Backward sequence
    features_backward = tf.reverse_sequence(features, tf.fill([batch_size], num_views), seq_axis=1)
    lstm_outputs_backward, _ = tf.nn.dynamic_rnn(multi_rnn_cell, features_backward, dtype=tf.float32, sequence_length=tf.fill([batch_size], num_views))
    lstm_outputs_backward = tf.reverse_sequence(lstm_outputs_backward, tf.fill([batch_size], num_views), seq_axis=1)
    # Element-wise maximum between forward and backward sequence
    lstm_outputs = tf.maximum(lstm_outputs_forward, lstm_outputs_backward)

    # Dropout to combat over-fitting
    lstm_outputs = tf.reshape(lstm_outputs, [batch_size * num_views, -1])
    fc = tf_util.fully_connected(lstm_outputs, num_features, activation_fn=leak_relu, scope='fully_connected', is_training=is_training)
    dp = tf_util.dropout(fc, is_training, scope='dropout', keep_prob=0.7)

    # Reshape tensor to fit 2D convolution
    dp = tf.reshape(dp, [batch_size * num_views, feature_height, feature_width, feature_channel])

    # Decoder
    edge_logits = get_decoder_network(dp, encoder, is_training)

    # Reshape into BxVxHxWx1
    edge_logits = tf.reshape(edge_logits, [batch_size, num_views, image_height, image_width, 1])

    return edge_logits

def get_loss(edge_logits, edges):
    weights = tf.where(tf.equal(edges, tf.zeros([])),\
                       tf.multiply(tf.reduce_sum(edges) / tf.to_float(tf.size(edges)), tf.ones_like(edges)),\
                       tf.multiply(1.0 - tf.reduce_sum(edges) / tf.to_float(tf.size(edges)), tf.ones_like(edges)))
    loss = tf.constant(1000.0, dtype=tf.float32) *\
           tf.reduce_mean(tf.multiply(weights, tf.nn.sigmoid_cross_entropy_with_logits(logits=edge_logits, labels=edges)))
    return loss


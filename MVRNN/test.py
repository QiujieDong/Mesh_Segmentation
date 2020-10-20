import argparse
import tensorflow as tf
import json
import numpy as np
import scipy.io
import glob
import os
import sys
from skimage import measure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import network as model

parser = argparse.ArgumentParser()
FLAGS = parser.parse_args()


# DEFAULT SETTINGS
gpu_to_use = 0
output_dir = os.path.join(BASE_DIR, './test_results')
output_verbose = True   # If true, output all color-coded part segmentation obj files

# MAIN SCRIPT
IMAGE_SIZE = 128 
NUM_VIEWS = 60
batch_size = 1              # DO NOT CHANGE

def get_file_name(file_path):
    parts = file_path.split('/')
    part = parts[-1]
    parts = part.split('.')
    return parts[0]

TESTING_FILE_LIST = [line.rstrip('\n') for line in open('./data/test.txt')]

def load_checkpoint(checkpoint_dir, session, var_list=None):
    print(' [*] Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    try:
        restorer = tf.train.Saver(var_list)
        restorer.restore(session, ckpt_path)
        print(' [*] Loading successful! Copy variables from % s' % ckpt_path)
        return True
    except:
        print(' [*] No suitable checkpoint!')
        return False

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def placeholder_inputs():
    images_ph = tf.placeholder(tf.float32, shape=(1, NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 1))
    edges_ph = tf.placeholder(tf.float32, shape=(1, NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 1))
    return images_ph, edges_ph

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict():
    is_training = False
    
    with tf.device('/gpu:'+str(gpu_to_use)):
        images_ph, edges_ph = placeholder_inputs()
        is_training_ph = tf.placeholder(tf.bool, shape=())

        # simple model
        edge_logits = model.get_model(images_ph, is_training=is_training_ph)
        loss = model.get_loss(edge_logits, edges_ph)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        flog = open(os.path.join(output_dir, 'log.txt'), 'w')

        # Restore variables from disk.
        ckpt_dir = './train_results/trained_models'
        if not load_checkpoint(ckpt_dir, sess):
            sess.run(tf.global_variables_initializer())

        if not os.path.exists('data/mv-rnn'):
            os.makedirs('data/mv-rnn')
        for l in range(len(TESTING_FILE_LIST)):
            images = np.zeros((1, NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 1))
            model_name = TESTING_FILE_LIST[l]
            if not os.path.exists('data/mv-rnn/' + model_name):
                os.makedirs('data/mv-rnn/' + model_name)
            for v in range(NUM_VIEWS):
                images[0, v, :, :, 0] = np.array(scipy.ndimage.imread('data/rgb/' + model_name + '/RGB-' + str(v).zfill(3) + '.png', mode = 'L'), dtype=np.float32)
            edge_logits_val = sess.run(edge_logits, feed_dict={images_ph: images, is_training_ph: is_training})
            edges = sigmoid(edge_logits_val)
            for v in range(NUM_VIEWS):
                scipy.misc.imsave('data/mv-rnn' + '/' + model_name + '/MV-RNN-' + str(v).zfill(3) + '.png', edges[0, v, :, :, 0])

            printout(flog, '[%2d/%2d] model %s' % ((l+1), len(TESTING_FILE_LIST), TESTING_FILE_LIST[l]))
            printout(flog, '----------')



with tf.Graph().as_default():
    predict()

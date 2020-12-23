import network as model
import argparse
import subprocess
import tensorflow as tf
import threading
import numpy as np
import tqdm
import scipy.io
import scipy.misc
import scipy.ndimage
import glob
from datetime import datetime
import json
import os
import sys
from skimage import feature
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=1,
                    help='Batch Size during training [default: 16]')
parser.add_argument('--epoch', type=int, default=5000,
                    help='Epoch to run [default: 50]')
parser.add_argument('--output_dir', type=str, default='train_results',
                    help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0,
                    help='Weight Decay [Default: 0.0]')
FLAGS = parser.parse_args()

# MAIN SCRIPT
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print('#### Batch Size: {0}'.format(batch_size))
IMAGE_SIZE = 128
print('### Image size: {0}'.format(IMAGE_SIZE))
NUM_VIEWS = 60
print('### Number of views: {0}'.format(NUM_VIEWS))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))

LEARNING_RATE = 0.0002
LEARNING_RATE_CLIP = 0.01
N_CRITIC = 5
TRAINING_EPOCHES = FLAGS.epoch
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))


def get_file_name(file_path):
    parts = file_path.split('/')
    part = parts[-1]
    parts = part.split('.')
    return parts[0]


TRAINING_FILE_LIST = [line.rstrip('\n') for line in open('./data/train.txt')]

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER = os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)


def printout(flog, data):
    print(data)
    flog.write(data + '\n')


def load_and_enqueue(sess, enqueue_op, images_ph, edges_ph):
    for epoch in range(100 * TRAINING_EPOCHES):
        train_file_idx = np.arange(0, len(TRAINING_FILE_LIST))
        np.random.shuffle(train_file_idx)
        for l in range(len(TRAINING_FILE_LIST)):
            images = np.zeros((NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 1))
            edges = np.zeros((NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 1))
            model_name = TRAINING_FILE_LIST[l]
            for v in range(NUM_VIEWS):
                images[v, :, :, 0] = np.array(scipy.ndimage.imread(
                    'data/rgb/' + model_name + '/RGB-' + str(v).zfill(3) + '.png', mode='L'), dtype=np.float32)
                E = scipy.ndimage.imread(
                    'data/groundtruth' + '/' + model_name + '/Output-' + str(v).zfill(3) + '.png', mode='L')
                E = np.where(E != 0, np.ones((IMAGE_SIZE, IMAGE_SIZE)),
                             np.zeros((IMAGE_SIZE, IMAGE_SIZE)))
                edges[v, :, :, 0] = E
            sess.run(enqueue_op, feed_dict={
                     images_ph: images, edges_ph: edges})


def placeholder_inputs():
    images_ph = tf.placeholder(tf.float32, shape=(
        NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 1))
    edges_ph = tf.placeholder(tf.float32, shape=(
        NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 1))
    return images_ph, edges_ph


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


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, target=None, args=None):
        super(StoppableThread, self).__init__(target=target, args=args)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            images_ph, edges_ph = placeholder_inputs()
            is_training_ph = tf.placeholder(tf.bool, shape=())

            queue = tf.FIFOQueue(capacity=10*batch_size, dtypes=[tf.float32, tf.float32],
                                 shapes=[[NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 1], [NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 1]])
            enqueue_op = queue.enqueue([images_ph, edges_ph])
            dequeue_images_ph, dequeue_edges_ph = queue.dequeue_many(
                batch_size)

            # model and loss
            edge_logits = model.get_model(
                dequeue_images_ph, is_training=is_training_ph)
            loss = model.get_loss(edge_logits, dequeue_edges_ph)

            # optimization
            total_var = tf.trainable_variables()
            train_step = tf.train.AdamOptimizer(
                learning_rate=LEARNING_RATE).minimize(loss, var_list=total_var)

        # write logs to the disk
        flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        ckpt_dir = './train_results/trained_models'
        if not load_checkpoint(ckpt_dir, sess):
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(
            SUMMARIES_FOLDER + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

        fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        def train_one_epoch(epoch_num):
            is_training = True

            num_data = len(TRAINING_FILE_LIST)
            num_batch = num_data // batch_size
            loss_acc = 0.0
            display_mark = max([num_batch // 4, 1])
            for i in tqdm.trange(num_batch):

                _, loss_val = sess.run([train_step, loss], feed_dict={
                                       is_training_ph: is_training})

                loss_acc += loss_val
                if ((i+1) % display_mark == 0):
                    printout(flog, 'Epoch %3d/%3d - Iter %4d/%d' %
                             (epoch_num+1, TRAINING_EPOCHES, i+1, num_batch))
                    printout(flog, 'total loss: %f' % (loss_acc / (i+1)))

            loss_acc = loss_acc * 1.0 / num_batch

            printout(flog, '\tMean total Loss: %f' % loss_acc)

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        coord = tf.train.Coordinator()
        for num_thread in range(4):
            t = StoppableThread(target=load_and_enqueue, args=(
                sess, enqueue_op, images_ph, edges_ph))
            t.setDaemon(True)
            t.start()
            coord.register_thread(t)

        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n>>> Training for the epoch %d/%d ...' %
                     (epoch+1, TRAINING_EPOCHES))

            train_one_epoch(epoch)

            if (epoch+1) % 1 == 0:
                cp_filename = saver.save(sess, os.path.join(
                    MODEL_STORAGE_PATH, 'epoch_' + str(epoch+1)+'.ckpt'))
                printout(
                    flog, 'Successfully store the checkpoint model into ' + cp_filename)

            flog.flush()
        flog.close()


if __name__ == '__main__':
    train()

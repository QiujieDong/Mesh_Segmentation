import sys, copy
from easydict import EasyDict
import json

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import rnn_model
import utils
import dataset


def calc_accuracy_test(dataset_expansion=False, logdir=None, labels=None, iter2use='last', classes_indices_to_use=None,
                       dnn_model=None, params=None, min_max_faces2use=[0, 4000], model_fn=None, n_walks_per_model=16, data_augmentation={}):
  # Prepare parameters for the evaluation
  if params is None:
    with open(logdir + '/params.txt') as fp:
      params = EasyDict(json.load(fp))
    if model_fn is not None:
      pass
    elif iter2use != 'last':
      model_fn = logdir + '/learned_model2keep--' + iter2use
      model_fn = model_fn.replace('//', '/')
    else:
      model_fn = tf.train.latest_checkpoint(logdir)
  else:
    params = copy.deepcopy(params)
  if logdir is not None:
    params.logdir = logdir
  params.batch_size = 1
  params.n_walks_per_model = n_walks_per_model
  params.classes_indices_to_use = None
  params.classes_indices_to_use = classes_indices_to_use

  # Prepare the dataset
  test_dataset, n_models_to_test = dataset.tf_mesh_dataset(params, dataset_expansion, mode=params.network_task,
                                                           shuffle_size=0, permute_file_names=True, min_max_faces2use=min_max_faces2use,
                                                           must_run_on_all=True, data_augmentation=data_augmentation)

  # If dnn_model is not provided, load it
  if dnn_model is None:
    dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim, model_fn,
                                     model_must_be_load=True, dump_model_visualization=False)

  n_pos_all = 0
  n_classes = 40
  all_confusion = np.zeros((n_classes, n_classes), dtype=np.int)
  pred_per_model_name = {}
  for i, data in tqdm(enumerate(test_dataset), total=n_models_to_test):
    name, ftrs, gt = data
    model_fn = name.numpy()[0].decode()
    model_name, n_faces = utils.get_model_name_from_npz_fn(model_fn)
    assert ftrs.shape[0] == 1, 'Must have one model per batch for test'
    ftrs = tf.reshape(ftrs, ftrs.shape[1:])
    gt = gt.numpy()[0]
    ftr2use = ftrs.numpy()
    predictions = dnn_model(ftr2use, classify=True, training=False).numpy()

    mean_pred = np.mean(predictions, axis=0)
    max_hit = np.argmax(mean_pred)

    if model_name not in pred_per_model_name.keys():
      pred_per_model_name[model_name] = [gt, np.zeros_like(mean_pred)]
    pred_per_model_name[model_name][1] += mean_pred

    all_confusion[int(gt), max_hit] += 1
    n_pos_all += (max_hit == gt)

  n_models = 0
  n_sucesses = 0
  all_confusion_all_faces = np.zeros((n_classes, n_classes), dtype=np.int)
  for k, v in pred_per_model_name.items():
    gt = v[0]
    pred = v[1]
    max_hit = np.argmax(pred)
    all_confusion_all_faces[gt, max_hit] += 1
    n_models += 1
    n_sucesses += max_hit == gt
  mean_accuracy_all_faces = n_sucesses / n_models

  # Print list of accuracy per model
  for confusion in [all_confusion, all_confusion_all_faces]:
    acc_per_class = []
    for i, name in enumerate(labels):
      this_type = confusion[i]
      n_this_type = this_type.sum()
      accuracy_this_type = this_type[i] / n_this_type
      if n_this_type:
        acc_per_class.append(accuracy_this_type)
      this_type_ = this_type.copy()
      this_type_[i] = -1
  mean_acc_per_class = np.mean(acc_per_class)

  return [mean_accuracy_all_faces, mean_acc_per_class], dnn_model


if __name__ == '__main__':
  from train_val import get_params
  utils.config_gpu(True)
  np.random.seed(0)
  tf.random.set_seed(0)

  if len(sys.argv) != 4:
    print('Use: python evaluate_classification.py <job> <part> <trained model directory>')
    print('For example: python evaluate_classification.py shrec11 10-10_A pretrained/0001-09.11.2020..19.57__shrec11_10-10_A')
  else:
    logdir = sys.argv[3]
    job = sys.argv[1]
    job_part = sys.argv[2]
    params = get_params(job, job_part)
    accs, _ = calc_accuracy_test(logdir=logdir, **params.full_accuracy_test)
    print('Mean accuracy:', accs[0])
    print('Mean per class accuracy:', accs[1])

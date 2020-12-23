import os

from easydict import EasyDict
import numpy as np

import utils
import dataset_prepare


def set_up_default_params(network_task, run_name, cont_run_number=0):
  '''
  Define dafault parameters, commonly for many test case
  '''
  params = EasyDict()

  params.cont_run_number = cont_run_number
  params.run_root_path = 'runs'
  params.logdir = utils.get_run_folder(params.run_root_path + '/', '__' + run_name, params.cont_run_number)
  params.model_fn = params.logdir + '/learned_model.keras'

  # Optimizer params
  params.optimizer_type = 'cycle'  # sgd / adam / cycle
  params.learning_rate_dynamics = 'cycle'
  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 1e-4,
                                    'step_size': 10000})
  params.n_models_per_test_epoch = 300
  params.gradient_clip_th = 1

  # Dataset params
  params.classes_indices_to_use = None
  params.train_dataset_size_limit = np.inf
  params.test_dataset_size_limit = np.inf
  params.network_task = network_task
  params.normalize_model = True
  params.sub_mean_for_data_augmentation = True
  params.datasets2use = {}
  params.test_data_augmentation = {}
  params.train_data_augmentation = {}

  params.network_tasks = [params.network_task]
  if params.network_task == 'classification':
    params.n_walks_per_model = 1
    params.one_label_per_model = True
    params.train_loss = ['cros_entr']
  elif params.network_task == 'semantic_segmentation':
    params.n_walks_per_model = 4
    params.one_label_per_model = False
    params.train_loss = ['cros_entr']
  else:
    raise Exception('Unsuported params.network_task: ' + params.network_task)
  params.batch_size = int(32 / params.n_walks_per_model)

  # Other params
  params.log_freq = 100
  params.walk_alg = 'random_global_jumps'
  params.net_input = ['dxdydz'] # 'xyz', 'dxdydz', 'jump_indication'
  params.train_min_max_faces2use = [0, np.inf]
  params.test_min_max_faces2use = [0, np.inf]

  params.net = 'RnnWalkNet'
  params.last_layer_actication = 'softmax'
  params.use_norm_layer = 'InstanceNorm' # BatchNorm / InstanceNorm / None
  params.layer_sizes = None

  params.initializers = 'orthogonal'
  params.adjust_vertical_model = False
  params.net_start_from_prev_net = None

  params.net_gru_dropout = 0
  params.uniform_starting_point = False

  params.full_accuracy_test = None

  params.iters_to_train = 60e3

  return params

# Classifications
# ---------------
def modelnet_params():
  params = set_up_default_params('classification', 'modelnet', 0)
  params.n_classes = 40

  p = 'modelnet40'
  params.train_min_max_faces2use = [0000, 4000]
  params.test_min_max_faces2use = [0000, 4000]

  ds_path = 'datasets_processed/modelnet40'
  params.datasets2use['train'] = [ds_path + '/*train*.npz']
  params.datasets2use['test'] = [ds_path + '/*test*.npz']

  params.seq_len = 800
  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                               'labels': dataset_prepare.model_net_labels,
                               'min_max_faces2use': params.test_min_max_faces2use,
                               'n_walks_per_model': 16 * 4,
                               }


  # Parameters to recheck:
  params.iters_to_train = 500e3
  params.net_input = ['xyz']

  return params

def cubes_params():
  params = set_up_default_params('classification', 'cubes', 0)
  params.n_classes = 22
  params.seq_len = 100
  params.min_seq_len = int(params.seq_len / 2)

  p = 'cubes'
  params.datasets2use['train'] = ['datasets_processed/' + p + '/*train*.npz']
  params.datasets2use['test'] = ['datasets_processed/' + p + '/*test*.npz']

  params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                               'labels': dataset_prepare.cubes_labels,
                               }

  params.iters_to_train = 460e3

  return params

def shrec11_params(split_part):
  # split_part is one of the following:
  # 10-10_A / 10-10_B / 10-10_C
  # 16-04_A / 16-04_B / 16-04_C
  params = set_up_default_params('classification', 'shrec11_' + split_part, 0)
  params.n_classes = 30
  params.seq_len = 100
  params.min_seq_len = int(params.seq_len / 2)

  params.datasets2use['train'] = ['datasets_processed/shrec11/' + split_part + '/train/*.npz']
  params.datasets2use['test']  = ['datasets_processed/shrec11/' + split_part + '/test/*.npz']

  params.train_data_augmentation = {'rotation': 360}

  params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                               'labels': dataset_prepare.shrec11_labels}

  return params


# Semantic Segmentation
# ---------------------
def human_seg_params():
  params = set_up_default_params('semantic_segmentation', 'human_seg', 0)
  params.n_classes = 9
  params.seq_len = 300
  params.min_seq_len = int(params.seq_len / 2)

  p = 'datasets_processed/human_seg_from_meshcnn/'
  params.datasets2use['train'] = [p + '*train*.npz']
  params.datasets2use['test']  = [p + '*test*.npz']

  params.train_data_augmentation = {'rotation': 360}

  params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                               'n_iters': 32}

  params.iters_to_train = 100e3

  return params


def coseg_params(type): # aliens / chairs / vases
  sub_folder = 'coseg_' + type
  p = 'datasets_processed/coseg_from_meshcnn/' + sub_folder + '/'
  params = set_up_default_params('semantic_segmentation', 'coseg_' + type, 0)
  params.n_classes = 10
  params.seq_len = 300
  params.min_seq_len = int(params.seq_len / 2)

  params.datasets2use['train'] = [p + '*train*.npz']
  params.datasets2use['test']  = [p + '*test*.npz']

  params.iters_to_train = 200e3
  params.train_data_augmentation = {'rotation': 360}

  params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                               'n_iters': 32}


  return params



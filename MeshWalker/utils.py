import os, shutil, psutil, json, copy
import datetime

import numpy as np
import tensorflow as tf

import evaluate_classification
import evaluate_segmentation


class color:
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'


def config_gpu(use_gpu=True):
  print('tf.__version__', tf.__version__)
  np.set_printoptions(suppress=True)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  try:
    if use_gpu:
      gpus = tf.config.experimental.list_physical_devices('GPU')
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  except:
    pass


def get_gpu_temprature():
  output = os.popen("nvidia-smi -q | grep 'GPU Current Temp' | cut -d' ' -f 24").read()
  output = ''.join(filter(str.isdigit, output))
  try:
    temp = int(output)
  except:
    temp = 0
  return temp


def backup_python_files_and_params(params):
  save_id = 0
  while 1:
    code_log_folder = params.logdir + '/.' + str(save_id)
    if not os.path.isdir(code_log_folder):
      os.makedirs(code_log_folder)
      for file in os.listdir():
        if file.endswith('py'):
          shutil.copyfile(file, code_log_folder + '/' + file)
      break
    else:
      save_id += 1

  # Dump params to text file
  try:
    prm2dump = copy.deepcopy(params)
    if 'hyper_params' in prm2dump.keys():
      prm2dump.hyper_params = str(prm2dump.hyper_params)
      prm2dump.hparams_metrics = prm2dump.hparams_metrics[0]._display_name
      for l in prm2dump.net:
        l['layer_function'] = 'layer_function'
    with open(params.logdir + '/params.txt', 'w') as fp:
      json.dump(prm2dump, fp, indent=2, sort_keys=True)
  except:
    pass


def get_run_folder(root_dir, str2add='', cont_run_number=False):
  try:
    all_runs = os.listdir(root_dir)
    run_ids = [int(d.split('-')[0]) for d in all_runs if '-' in d]
    if cont_run_number:
      n = [i for i, m in enumerate(run_ids) if m == cont_run_number][0]
      run_dir = root_dir + all_runs[n]
      print('Continue to run at:', run_dir)
      return run_dir
    n = np.sort(run_ids)[-1]
  except:
    n = 0
  now = datetime.datetime.now()
  return root_dir + str(n + 1).zfill(4) + '-' + now.strftime("%d.%m.%Y..%H.%M") + str2add


last_free_mem = np.inf
def check_mem_and_exit_if_full():
  global last_free_mem
  free_mem = psutil.virtual_memory().available + psutil.swap_memory().free
  free_mem_gb = round(free_mem / 1024 / 1024 / 1024, 2)
  if last_free_mem > free_mem_gb + 0.25:
    last_free_mem = free_mem_gb
    print('free_mem', free_mem_gb, 'GB')
  if free_mem_gb < 1:
    print('!!! Exiting due to memory full !!!')
    exit(111)
  return free_mem_gb


next_iter_to_keep = 0 # Should be set by -train_val- function, each time job starts
def save_model_if_needed(iterations, dnn_model, params):
  global next_iter_to_keep
  iter_th = 20000
  keep = iterations.numpy() >= next_iter_to_keep
  dnn_model.save_weights(params.logdir, iterations.numpy(), keep=keep)
  if keep:
    if iterations < iter_th:
      next_iter_to_keep = iterations * 2
    else:
      next_iter_to_keep = int(iterations / iter_th) * iter_th + iter_th
    if params.full_accuracy_test is not None:
      if params.network_task == 'semantic_segmentation':
        accuracy, _ = evaluate_segmentation.calc_accuracy_test(params=params, dnn_model=dnn_model, **params.full_accuracy_test)
      elif params.network_task == 'classification':
        accuracy, _ = evaluate_classification.calc_accuracy_test(params=params, dnn_model=dnn_model, **params.full_accuracy_test)
      with open(params.logdir + '/log.txt', 'at') as f:
        f.write('Accuracy: ' + str(np.round(np.array(accuracy) * 100, 2)) + '%, Iter: ' + str(iterations.numpy()) + '\n')
      tf.summary.scalar('full_accuracy_test/overall', accuracy[0], step=iterations)
      tf.summary.scalar('full_accuracy_test/mean', accuracy[1], step=iterations)


def get_dataset_type_from_name(tf_names):
  name_str = tf_names[0].numpy().decode()
  return name_str[:name_str.find(':')]


def get_model_name_from_npz_fn(npz_fn):
  fn = npz_fn.split('/')[-1].split('.')[-2]
  sp_fn = fn.split('_')
  if npz_fn.find('/shrec11') == -1:
    sp_fn = sp_fn[1:]
  i = np.where([s.isdigit() for s in sp_fn])[0][0]
  model_name = '_'.join(sp_fn[:i + 1])
  n_faces = int(sp_fn[-1])

  return model_name, n_faces





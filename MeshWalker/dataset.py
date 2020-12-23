import glob, os, copy

import tensorflow as tf
import numpy as np

import utils
import walks
import dataset_prepare


# Glabal list of dataset parameters. Used as part of runtime acceleration affort.
dataset_params_list = []

# ------------------------------------------------------------------ #
# ---------- Some utility functions -------------------------------- #
# ------------------------------------------------------------------ #
def load_model_from_npz(npz_fn):
  if npz_fn.find(':') != -1:
    npz_fn = npz_fn.split(':')[1]
  mesh_data = np.load(npz_fn, encoding='latin1', allow_pickle=True)
  return mesh_data


def norm_model(vertices):
  # Move the model so the bbox center will be at (0, 0, 0)
  mean = np.mean((np.min(vertices, axis=0), np.max(vertices, axis=0)), axis=0)
  vertices -= mean

  # Scale model to fit into the unit ball
  norm_with = np.max(vertices)
  vertices /= norm_with

  if norm_model.sub_mean_for_data_augmentation:     # TODO: check again
    vertices -= np.nanmean(vertices, axis=0)


def get_file_names(pathname_expansion, min_max_faces2use):
  filenames_ = glob.glob(pathname_expansion)
  filenames = []
  for fn in filenames_:
    try:
      n_faces = int(fn.split('.')[-2].split('_')[-1])
      if n_faces > min_max_faces2use[1] or n_faces < min_max_faces2use[0]:
        continue
    except:
      pass
    filenames.append(fn)
  assert len(filenames) > 0, 'DATASET error: no files in directory to be used! \nDataset directory: ' + pathname_expansion

  return filenames


def dump_all_fns_to_file(filenames, params):
  if 'logdir' in params.keys():
    for n in range(10):
      log_fn = params.logdir + '/dataset_files_' + str(n).zfill(2) + '.txt'
      if not os.path.isfile(log_fn):
        try:
          with open(log_fn, 'w') as f:
            for fn in filenames:
              f.write(fn + '\n')
        except:
          pass
        break


def filter_fn_by_class(filenames_, classes_indices_to_use):
  filenames = []
  for fn in filenames_:
    mesh_data = np.load(fn, encoding='latin1', allow_pickle=True)
    if classes_indices_to_use is not None and mesh_data['label'] not in classes_indices_to_use:
      continue
    filenames.append(fn)
  return filenames


def data_augmentation_rotation(vertices):
  max_rot_ang_deg = data_augmentation_rotation.max_rot_ang_deg
  x = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
  y = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
  z = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
  A = np.array(((np.cos(x), -np.sin(x), 0),
                (np.sin(x), np.cos(x), 0),
                (0, 0, 1)),
               dtype=vertices.dtype)
  B = np.array(((np.cos(y), 0, -np.sin(y)),
                (0, 1, 0),
                (np.sin(y), 0, np.cos(y))),
               dtype=vertices.dtype)
  C = np.array(((1, 0, 0),
                (0, np.cos(z), -np.sin(z)),
                (0, np.sin(z), np.cos(z))),
               dtype=vertices.dtype)
  np.dot(vertices, A, out=vertices)
  np.dot(vertices, B, out=vertices)
  np.dot(vertices, C, out=vertices)

# ------------------------------------------------------------------ #
# --- Some functions used to set up the RNN input "features" ------- #
# ------------------------------------------------------------------ #
def fill_xyz_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = vertices[seq[1:seq_len + 1]]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx


def fill_dxdydz_features(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = np.diff(vertices[seq[:seq_len + 1]], axis=0) * 100
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx


def fill_vertex_indices(features, f_idx, vertices, mesh_extra, seq, jumps, seq_len):
  walk = seq[1:seq_len + 1][:, None]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 1
  return f_idx


# ------------------------------------------------------------------ #
def setup_data_augmentation(dataset_params, data_augmentation):
  dataset_params.data_augmentaion_vertices_functions = []
  if 'rotation' in data_augmentation.keys() and data_augmentation['rotation']:
    data_augmentation_rotation.max_rot_ang_deg = data_augmentation['rotation']
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_rotation)


def setup_features_params(dataset_params, params):
  if params.uniform_starting_point:
    dataset_params.area = 'all'
  else:
    dataset_params.area = -1
  norm_model.sub_mean_for_data_augmentation = params.sub_mean_for_data_augmentation
  dataset_params.support_mesh_cnn_ftrs = False
  dataset_params.fill_features_functions = []
  dataset_params.number_of_features = 0
  net_input = params.net_input
  if 'xyz' in net_input:
    dataset_params.fill_features_functions.append(fill_xyz_features)
    dataset_params.number_of_features += 3
  if 'dxdydz' in net_input:
    dataset_params.fill_features_functions.append(fill_dxdydz_features)
    dataset_params.number_of_features += 3
  if 'vertex_indices' in net_input:
    dataset_params.fill_features_functions.append(fill_vertex_indices)
    dataset_params.number_of_features += 1

  dataset_params.edges_needed = True
  if params.walk_alg == 'random_global_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_random_global_jumps
  else:
    raise Exception('Walk alg not recognized: ' + params.walk_alg)

  return dataset_params.number_of_features


# ------------------------------------------------- #
# ------- TensorFlow dataset functions ------------ #
# ------------------------------------------------- #
def generate_walk_py_fun(fn, vertices, faces, edges, labels, params_idx):
  return tf.py_function(
    generate_walk,
    inp=(fn, vertices, faces, edges, labels, params_idx),
    Tout=(fn.dtype, vertices.dtype, tf.int32)
  )


def generate_walk(fn, vertices, faces, edges, labels_from_npz, params_idx):
  mesh_data = {'vertices': vertices.numpy(),
               'faces': faces.numpy(),
               'edges': edges.numpy(),
               }
  if dataset_params_list[params_idx[0]].label_per_step:
    mesh_data['labels'] = labels_from_npz.numpy()

  dataset_params = dataset_params_list[params_idx[0].numpy()]
  features, labels = mesh_data_to_walk_features(mesh_data, dataset_params)

  if dataset_params_list[params_idx[0]].label_per_step:
    labels_return = labels
  else:
    labels_return = labels_from_npz

  return fn[0], features, labels_return


def mesh_data_to_walk_features(mesh_data, dataset_params):
  vertices = mesh_data['vertices']
  seq_len = dataset_params.seq_len
  if dataset_params.set_seq_len_by_n_faces:
    seq_len = int(mesh_data['vertices'].shape[0])
    seq_len = min(seq_len, dataset_params.seq_len)

  # Preprocessing
  if dataset_params.adjust_vertical_model:
    vertices[:, 1] -= vertices[:, 1].min()
  if dataset_params.normalize_model:
    norm_model(vertices)

  # Data augmentation
  for data_augmentaion_function in dataset_params.data_augmentaion_vertices_functions:
    data_augmentaion_function(vertices)

  # Get essential data from file
  if dataset_params.label_per_step:
    mesh_labels = mesh_data['labels']
  else:
    mesh_labels = -1 * np.ones((vertices.shape[0],))

  mesh_extra = {}
  mesh_extra['n_vertices'] = vertices.shape[0]
  if dataset_params.edges_needed:
    mesh_extra['edges'] = mesh_data['edges']

  features = np.zeros((dataset_params.n_walks_per_model, seq_len, dataset_params.number_of_features), dtype=np.float32)
  labels   = np.zeros((dataset_params.n_walks_per_model, seq_len), dtype=np.int32)

  for walk_id in range(dataset_params.n_walks_per_model):
    f0 = np.random.randint(vertices.shape[0])                               # Get walk starting point
    seq, jumps = dataset_params.walk_function(mesh_extra, f0, seq_len)      # Get walk indices (and jump indications)

    f_idx = 0
    for fill_ftr_fun in dataset_params.fill_features_functions:
      f_idx = fill_ftr_fun(features[walk_id], f_idx, vertices, mesh_extra, seq, jumps, seq_len)
    if dataset_params.label_per_step:
      labels[walk_id] = mesh_labels[seq[1:seq_len + 1]]

  return features, labels


def setup_dataset_params(params, data_augmentation):
  p_idx = len(dataset_params_list)
  ds_params = copy.deepcopy(params)
  ds_params.set_seq_len_by_n_faces = False

  setup_data_augmentation(ds_params, data_augmentation)
  setup_features_params(ds_params, params)

  dataset_params_list.append(ds_params)

  return p_idx


class OpenMeshDataset(tf.data.Dataset):
  # OUTPUT:      (fn,               vertices,          faces,           edges,           labels,          params_idx)
  OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int16, tf.dtypes.int16, tf.dtypes.int32, tf.dtypes.int16)

  def _generator(fn_, params_idx):
    fn = fn_[0]
    with np.load(fn, encoding='latin1', allow_pickle=True) as mesh_data:
      vertices = mesh_data['vertices']
      faces = mesh_data['faces']
      edges = mesh_data['edges']
      if dataset_params_list[params_idx].label_per_step:
        labels = mesh_data['labels']
      else:
        labels = mesh_data['label']

      name = mesh_data['dataset_name'].tolist() + ':' + fn.decode()

    yield ([name], vertices, faces, edges, labels, [params_idx])

  def __new__(cls, filenames, params_idx):
    return tf.data.Dataset.from_generator(
      cls._generator,
      output_types=cls.OUTPUT_TYPES,
      args=(filenames, params_idx)
    )


def tf_mesh_dataset(params, pathname_expansion, mode=None, size_limit=np.inf, shuffle_size=1000,
                    permute_file_names=True, min_max_faces2use=[0, np.inf], data_augmentation={},
                    must_run_on_all=False, min_dataset_size=16):
  params_idx = setup_dataset_params(params, data_augmentation)
  number_of_features = dataset_params_list[params_idx].number_of_features
  params.net_input_dim = number_of_features
  mesh_data_to_walk_features.SET_SEED_WALK = 0

  filenames = get_file_names(pathname_expansion, min_max_faces2use)
  if params.classes_indices_to_use is not None:
    filenames = filter_fn_by_class(filenames, params.classes_indices_to_use)

  if permute_file_names:
    filenames = np.random.permutation(filenames)
  else:
    filenames.sort()
    filenames = np.array(filenames)

  if size_limit < len(filenames):
    filenames = filenames[:size_limit]
  n_items = len(filenames)
  if len(filenames) < min_dataset_size:
    filenames = filenames.tolist() * (int(min_dataset_size / len(filenames)) + 1)

  if mode == 'classification':
    dataset_params_list[params_idx].label_per_step = False
  elif mode == 'semantic_segmentation':
    dataset_params_list[params_idx].label_per_step = True
  else:
    raise Exception('DS mode ?')

  dump_all_fns_to_file(filenames, params)

  def _open_npz_fn(*args):
    return OpenMeshDataset(args, params_idx)

  ds = tf.data.Dataset.from_tensor_slices(filenames)
  if shuffle_size:
    ds = ds.shuffle(shuffle_size)
  ds = ds.interleave(_open_npz_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.cache()
  ds = ds.map(generate_walk_py_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(params.batch_size, drop_remainder=False)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  return ds, n_items

if __name__ == '__main__':
  utils.config_gpu(False)
  np.random.seed(1)

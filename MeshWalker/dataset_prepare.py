import glob, os, shutil, sys, json
from pathlib import Path

import pylab as plt
import trimesh
import open3d
from easydict import EasyDict
import numpy as np
from tqdm import tqdm

import utils


# Labels for all datasets
# -----------------------
sigg17_part_labels = ['---', 'head', 'hand', 'lower-arm', 'upper-arm', 'body', 'upper-lag', 'lower-leg', 'foot']
sigg17_shape2label = {v: k for k, v in enumerate(sigg17_part_labels)}

model_net_labels = [
  'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
  'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
  'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
  'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
]
model_net_shape2label = {v: k for k, v in enumerate(model_net_labels)}

cubes_labels = [
  'apple',  'bat',      'bell',     'brick',      'camel',
  'car',    'carriage', 'chopper',  'elephant',   'fork',
  'guitar', 'hammer',   'heart',    'horseshoe',  'key',
  'lmfish', 'octopus',  'shoe',     'spoon',      'tree',
  'turtle', 'watch'
]
cubes_shape2label = {v: k for k, v in enumerate(cubes_labels)}

shrec11_labels = [
  'armadillo',  'man',      'centaur',    'dinosaur',   'dog2',
  'ants',       'rabbit',   'dog1',       'snake',      'bird2',
  'shark',      'dino_ske', 'laptop',     'santa',      'flamingo',
  'horse',      'hand',     'lamp',       'two_balls',  'gorilla',
  'alien',      'octopus',  'cat',        'woman',      'spiders',
  'camel',      'pliers',   'myScissor',  'glasses',    'bird1'
]
shrec11_shape2label = {v: k for k, v in enumerate(shrec11_labels)}

coseg_labels = [
  '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c',
]
coseg_shape2label = {v: k for k, v in enumerate(coseg_labels)}


def calc_mesh_area(mesh):
  t_mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'], process=False)
  mesh['area_faces'] = t_mesh.area_faces
  mesh['area_vertices'] = np.zeros((mesh['vertices'].shape[0]))
  for f_index, f in enumerate(mesh['faces']):
    for v in f:
      mesh['area_vertices'][v] += mesh['area_faces'][f_index] / f.size


def prepare_edges_and_kdtree(mesh):
  vertices = mesh['vertices']
  faces = mesh['faces']
  mesh['edges'] = [set() for _ in range(vertices.shape[0])]
  for i in range(faces.shape[0]):
    for v in faces[i]:
      mesh['edges'][v] |= set(faces[i])
  for i in range(vertices.shape[0]):
    if i in mesh['edges'][i]:
      mesh['edges'][i].remove(i)
    mesh['edges'][i] = list(mesh['edges'][i])
  max_vertex_degree = np.max([len(e) for e in mesh['edges']])
  for i in range(vertices.shape[0]):
    if len(mesh['edges'][i]) < max_vertex_degree:
      mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
  mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)

  mesh['kdtree_query'] = []
  t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  n_nbrs = min(10, vertices.shape[0] - 2)
  for n in range(vertices.shape[0]):
    d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
    i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
    mesh['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  mesh['kdtree_query'] = np.array(mesh['kdtree_query'])
  assert mesh['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(mesh['kdtree_query'].shape[1])


def add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, dataset_name, dump_model=True):
  m = {}
  for k, v in mesh_data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels':
        m[field] = np.zeros((0,))
      if field == 'dataset_name':
        m[field] = dataset_name
      if field == 'walk_cache':
        m[field] = np.zeros((0,))
      if field == 'kdtree_query' or field == 'edges':
        prepare_edges_and_kdtree(m)

  if dump_model:
    np.savez(out_fn, **m)

  return m


def get_labels(dataset_name, mesh, file, fn2labels_map=None):
  v_labels_fuzzy = np.zeros((0,))
  if dataset_name.startswith('coseg') or dataset_name == 'human_seg_from_meshcnn':
    labels_fn = '/'.join(file.split('/')[:-2]) + '/seg/' + file.split('/')[-1].split('.')[-2] + '.eseg'
    e_labels = np.loadtxt(labels_fn)
    v_labels = [[] for _ in range(mesh['vertices'].shape[0])]
    faces = mesh['faces']

    fuzzy_labels_fn = '/'.join(file.split('/')[:-2]) + '/sseg/' + file.split('/')[-1].split('.')[-2] + '.seseg'
    seseg_labels = np.loadtxt(fuzzy_labels_fn)
    v_labels_fuzzy = np.zeros((mesh['vertices'].shape[0], seseg_labels.shape[1]))

    edge2key = dict()
    edges = []
    edges_count = 0
    for face_id, face in enumerate(faces):
      faces_edges = []
      for i in range(3):
        cur_edge = (face[i], face[(i + 1) % 3])
        faces_edges.append(cur_edge)
      for idx, edge in enumerate(faces_edges):
        edge = tuple(sorted(list(edge)))
        faces_edges[idx] = edge
        if edge not in edge2key:
          v_labels_fuzzy[edge[0]] += seseg_labels[edges_count]
          v_labels_fuzzy[edge[1]] += seseg_labels[edges_count]

          edge2key[edge] = edges_count
          edges.append(list(edge))
          v_labels[edge[0]].append(e_labels[edges_count])
          v_labels[edge[1]].append(e_labels[edges_count])
          edges_count += 1

    assert np.max(np.sum(v_labels_fuzzy != 0, axis=1)) <= 3, 'Number of non-zero labels must not acceeds 3!'

    vertex_labels = []
    for l in v_labels:
      l2add = np.argmax(np.bincount(l))
      vertex_labels.append(l2add)
    vertex_labels = np.array(vertex_labels)
    model_label = np.zeros((0,))

    return model_label, vertex_labels, v_labels_fuzzy
  else:
    tmp = file.split('/')[-1]
    model_name = '_'.join(tmp.split('_')[:-1])
    if dataset_name.lower().startswith('modelnet'):
      model_label = model_net_shape2label[model_name]
    elif dataset_name.lower().startswith('cubes'):
      model_label = cubes_shape2label[model_name]
    elif dataset_name.lower().startswith('shrec11'):
      model_name = file.split('/')[-3]
      if fn2labels_map is None:
        model_label = shrec11_shape2label[model_name]
      else:
        file_index = int(file.split('.')[-2].split('T')[-1])
        model_label = fn2labels_map[file_index]
    else:
      raise Exception('Cannot find labels for the dataset')
    vertex_labels = np.zeros((0,))
    return model_label, vertex_labels, v_labels_fuzzy


def remesh(mesh_orig, target_n_faces, add_labels=False, labels_orig=None):
  labels = labels_orig
  if target_n_faces < np.asarray(mesh_orig.triangles).shape[0]:
    mesh = mesh_orig.simplify_quadric_decimation(target_n_faces)
    str_to_add = '_simplified_to_' + str(target_n_faces)
    mesh = mesh.remove_unreferenced_vertices()
    if add_labels and labels_orig.size:
      labels = fix_labels_by_dist(np.asarray(mesh.vertices), np.asarray(mesh_orig.vertices), labels_orig)
  else:
    mesh = mesh_orig
    str_to_add = '_not_changed_' + str(np.asarray(mesh_orig.triangles).shape[0])

  return mesh, labels, str_to_add


def load_mesh(model_fn, classification=True):
  # To load and clean up mesh - "remove vertices that share position"
  if classification:
    mesh_ = trimesh.load_mesh(model_fn, process=True)
    mesh_.remove_duplicate_faces()
  else:
    mesh_ = trimesh.load_mesh(model_fn, process=False)
  mesh = open3d.geometry.TriangleMesh()
  mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
  mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)

  return mesh

def create_tmp_dataset(model_fn, p_out, n_target_faces):
  fileds_needed = ['vertices', 'faces', 'edge_features', 'edges_map', 'edges', 'kdtree_query',
                   'label', 'labels', 'dataset_name']
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  mesh_orig = load_mesh(model_fn)
  mesh, labels, str_to_add = remesh(mesh_orig, n_target_faces)
  labels = np.zeros((np.asarray(mesh.vertices).shape[0],), dtype=np.int16)
  mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': 0, 'labels': labels})
  out_fn = p_out + '/tmp'
  add_fields_and_dump_model(mesh_data, fileds_needed, out_fn, 'tmp')


def prepare_directory(dataset_name, pathname_expansion=None, p_out=None, n_target_faces=None, add_labels=True,
                                   size_limit=np.inf, fn_prefix='', verbose=True, classification=True):
  fileds_needed = ['vertices', 'faces', 'edges',
                   'label', 'labels', 'dataset_name', 'labels_fuzzy']

  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  filenames = glob.glob(pathname_expansion)
  filenames.sort()
  if len(filenames) > size_limit:
    filenames = filenames[:size_limit]
  for file in tqdm(filenames, disable=1 - verbose):
    out_fn = p_out + '/' + fn_prefix + os.path.split(file)[1].split('.')[0]
    mesh = load_mesh(file, classification=classification)
    mesh_orig = mesh
    mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})
    if add_labels:
      if type(add_labels) is list:
        fn2labels_map = add_labels
      else:
        fn2labels_map = None
      label, labels_orig, v_labels_fuzzy = get_labels(dataset_name, mesh_data, file, fn2labels_map=fn2labels_map)
    else:
      label = np.zeros((0, ))
    for this_target_n_faces in n_target_faces:
      mesh, labels, str_to_add = remesh(mesh_orig, this_target_n_faces, add_labels=add_labels, labels_orig=labels_orig)
      mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles), 'label': label, 'labels': labels})
      mesh_data['labels_fuzzy'] = v_labels_fuzzy
      out_fc_full = out_fn + str_to_add
      add_fields_and_dump_model(mesh_data, fileds_needed, out_fc_full, dataset_name)

# ------------------------------------------------------- #

def prepare_modelnet40():
  n_target_faces = [1000, 2000, 4000]
  labels2use = model_net_labels
  for i, name in tqdm(enumerate(labels2use)):
    for part in ['test', 'train']:
      pin = 'datasets_raw/ModelNet40/' + name + '/' + part + '/'
      p_out = 'datasets_processed/modelnet40/'
      prepare_directory('modelnet40', pathname_expansion=pin + '*.off',
                        p_out=p_out, add_labels='modelnet', n_target_faces=n_target_faces,
                        fn_prefix=part + '_', verbose=False)


def prepare_cubes(labels2use=cubes_labels,
                  path_in='datasets_raw/from_meshcnn/cubes/',
                  p_out='datasets_processed/cubes'):
  dataset_name = 'cubes'
  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  for i, name in enumerate(labels2use):
    print('-->>>', name)
    for part in ['test', 'train']:
      pin = path_in + name + '/' + part + '/'
      prepare_directory(dataset_name, pathname_expansion=pin + '*.obj',
                                     p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                                     classification=False)


def prepare_seg_from_meshcnn(dataset, subfolder=None):
  if dataset == 'human_body':
    dataset_name = 'human_seg_from_meshcnn'
    p_in2add = 'human_seg'
    p_out_sub = p_in2add
    p_ext = ''
  elif dataset == 'coseg':
    p_out_sub = dataset_name = 'coseg'
    p_in2add = dataset_name + '/' + subfolder
    p_ext = subfolder

  path_in = 'datasets_raw/from_meshcnn/' + p_in2add + '/'
  p_out = 'datasets_processed/' + p_out_sub + '_from_meshcnn/' + p_ext

  for part in ['test', 'train']:
    pin = path_in + '/' + part + '/'
    prepare_directory(dataset_name, pathname_expansion=pin + '*.obj',
                      p_out=p_out, add_labels=dataset_name, fn_prefix=part + '_', n_target_faces=[np.inf],
                      classification=False)


# ------------------------------------------------------- #


def prepare_one_dataset(dataset_name):
  dataset_name = dataset_name.lower()
  if dataset_name == 'modelnet40' or dataset_name == 'modelnet':
    prepare_modelnet40()

  if dataset_name == 'shrec11':
    print('To do later')

  if dataset_name == 'cubes':
    prepare_cubes()

  # Semantic Segmentations
  if dataset_name == 'human_seg':
    prepare_seg_from_meshcnn('human_body')

  if dataset_name == 'coseg':
    prepare_seg_from_meshcnn('coseg', 'coseg_aliens')
    prepare_seg_from_meshcnn('coseg', 'coseg_chairs')
    prepare_seg_from_meshcnn('coseg', 'coseg_vases')


if __name__ == '__main__':
  utils.config_gpu(False)
  np.random.seed(1)

  if len(sys.argv) != 2:
    print('Use: python dataset_prepare.py <dataset name>')
    print('For example: python dataset_prepare.py cubes')
    print('Another example: python dataset_prepare.py all')
  else:
    dataset_name = sys.argv[1]
    if dataset_name == 'all':
      for dataset_name in ['cubes', 'human_seg', 'coseg', 'modelnet40']:
        prepare_one_dataset(dataset_name)
    else:
      prepare_one_dataset(dataset_name)


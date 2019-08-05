
"""Utility functions for dealing with Drosophila melanogaster data."""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pickle import load

import cameras
import viz
import h5py
import glob
import copy
import sys
import signal_utils
import procrustes

DATA_DIR = "flydata/"
FILES = [os.path.join(DATA_DIR, f) \
     for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
FILES.sort(reverse=False)
FILE_NUM = len(FILES)
FILE_BAD = FILES[0]
FILE_REF = FILES[FILE_NUM-1]

TRAIN_FILES = FILES[:2]
TEST_FILES = FILES[2:]

CAMERA_TO_USE = 1
CAMERA_PROJ = 1

DIMENSIONS = 38
SUPERIMP_POSITION = 0
ROOT_POSITION = 0
BODY_COXA = [0, 5, 10, 19, 24, 29]
DIMENSIONS_TO_USE = list(range(15))

def read_data(dir):
  with (open(dir, "rb")) as file:
    try:
      return load(file)
    except EOFError:
      return None

def load_data(dim, rcams=None, camera_frame=False, superimp=False, changeorig=False, procrustes=False,
  lowpass=False):
  """
  Loads 2d ground truth from disk, and puts it in an easy-to-acess dictionary

  Args
    dim: Integer={2,3}. Load 2 or 3-dimensional data
    rcams:
    camera_frame:
    superimp: whether to superimpose the data from different files
    changeorig: whether to change origin of the system to ROOT_POSITION
    proc_gt:
    lowpass:
  Returns:
    data: Dictionary with keys k=(subject, 0)
      values: matrix (38, 3) with the 3d points data
  """

  dics = []
  dims = np.zeros((FILE_NUM+1), dtype=int)
  for idx, f in enumerate(FILES):
    print("[*] reading file {0}".format(f))
    if dim == 3:
      d = read_data(f)['points3d']
      if camera_frame:
        d = transform_world_to_camera(d, rcams, f)
      dics.append(d)
      dims[idx+1] = dims[idx] + d.shape[0]
    else: # dim == 2
      d = read_data(f)['points2d'][CAMERA_TO_USE]
      dics.append(d)
      dims[idx+1] = dims[idx] + d.shape[0]
  end = dics[0].shape[0]
  d_data = np.vstack(dics)
  print("[+] done reading data, shape: ", d_data.shape)
  
  if dim == 3 and lowpass:
    d_data = signal_utils.filter_batch(d_data) 
  if dim == 3 and changeorig:
    d_data = change_origin(d_data)
  
  data = {}
  for idx, f in enumerate(FILES):
    data[(f)] = d_data[dims[idx]:dims[idx+1]]
  
  if dim == 3 and procrustes:
    data = apply_procrustes(data)
  if dim == 3 and superimp:
    data = superimpose(data)

  return data

def transform_world_to_camera(t3d_world, rcams, f):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args
      t3d_world: matrix Nx with 3d poses in world coordinates
      rcams: dictionary with cameras
    Return:
      t3d_camera: dictionary with keys (subject, camera)
        with 3d poses in camera coordinate
    """
    t3d_camera = np.zeros(t3d_world.shape)
    R, T, _, _, _, intr = rcams[ (f, CAMERA_PROJ) ]
    for i in range(t3d_world.shape[0]):
      camera_coord = cameras.world_to_camera_frame( t3d_world[i], R, T, intr )
      t3d_camera[i] = camera_coord

    return t3d_camera

def apply_procrustes(data3d):
  ground_truth = data3d[(FILE_REF)]
  for f in range(FILES):
    if f == FILE_REF:
      continue
    pts_t, d = procrustes.procrustes(data3d[(f)], ground_truth, False)
    data3d[(f)] = pts_t
 
def superimpose(data3d):
  tmp = data3d.copy()
  tmp_bad = tmp.pop((FILE_BAD), None)
  tmp = np.vstack( list(tmp.values()) )
  for coord in range(3):
    dist = np.mean(tmp[:,SUPERIMP_POSITION,coord]) -\
       np.mean(tmp_bad[:,SUPERIMP_POSITION,coord])
    for (f) in data3d.keys():
      if f is FILE_BAD:
        data3d[(f)][:,:,coord] += dist
  return data3d

def change_origin(data3d):
  for i in range(data3d.shape[0]):
    for coord in range(3):
      data3d[i,:,coord] -= data3d[i,ROOT_POSITION,coord]
  return data3d

def split_train_test(data, files, dim, camera_frame=False):
  dic = {}
  for f in files:
    if dim == 3 and not camera_frame:
      dic[ (f, 0) ] = data[(f)]
    elif dim == 3 and camera_frame:
      dic[ (f, CAMERA_PROJ) ] = data[(f)]
    else: # dim == 2
      dic[ (f, CAMERA_TO_USE) ] = data[(f)]     
  return dic

def normalization_stats(complete_data, changeorig, dim):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    complete_data: nxd np array with poses
    changeorig: do we have a root position?
    dim: integer={2,3} dimensionality of the data
  Returns
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions not used in the model
    dimensions_to_use: list of dimensions used in the model
  """
  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data_mean = np.mean(complete_data, axis=0)
  data_std = np.std(complete_data, axis=0)

  if changeorig:
    dtu = [x for x in DIMENSIONS_TO_USE if x != ROOT_POSITION]
  else:
    dtu = DIMENSIONS_TO_USE
  
  dimensions_to_ignore = []
  if dim == 2:
    dimensions_to_use = np.sort( np.hstack((np.array(dtu)*2, np.array(dtu)*2+1)) )
    dimensions_to_ignore = np.delete(np.arange(DIMENSIONS*2), dimensions_to_use)
  else: # dim == 3
    dimensions_to_use = np.sort( np.hstack((np.array(dtu)*3, np.array(dtu)*3+1, np.array(dtu)*3+2)) )
    dimensions_to_ignore = np.delete(np.arange(DIMENSIONS*3), dimensions_to_use)
  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use

def normalize_data(data, data_mean, data_std, dim_to_use, dim):
  """
  Normalizes a dictionary of poses

  Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions to keep in the data
    dim. integer={2,3} dimensionality of the data
  Returns
    data_out: dictionary with same keys as data, but values have been normalized
  """
  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data_out = {}
  
  for key in data.keys():
    data_out[ key ] = np.reshape(data[ key ], (-1, dim*DIMENSIONS))
    data_out[ key ][:, dim_to_use] = \
      np.divide( (data_out[key][:, dim_to_use] - data_mean[dim_to_use]), data_std[dim_to_use] )
  
  return data_out

def unNormalize_dic(data, data_mean, data_std, dim_to_use):
  """
  unNormalizes a dictionary of poses

  Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions to keep in the data
  Returns
    data_out: dictionary with same keys as data, but values have been unNormalized
  """
  data_out = {}

  for key in data.keys():
    data_out[key] = np.copy(data[key])
    data_out[ key ][:, dim_to_use] = \
       np.multiply( data_out[key][:, dim_to_use], data_std[dim_to_use] ) + data_mean[dim_to_use]

  return data_out

def unNormalize_batch(normalized_data, data_mean, data_std, dim_to_use):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing

  Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions that are used in the original data
  Returns
    orig_data: the input normalized_data, but unnormalized
  """
  T = normalized_data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality

  orig_data = np.zeros((T, D), dtype=np.float32)
  orig_data[:, dim_to_use] = normalized_data

  # Multiply times stdev and add the mean
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  orig_data = np.multiply(orig_data, stdMat) + meanMat
  
  return orig_data

def project_to_cameras(poses_set, cams, ncams=4):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    ncams: number of cameras per subject
  Returns
    t2d: dictionary with 2d poses
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    for cam in range( ncams ):
      R, T, f, c, k, p, name = cams[ (subj, cam+1) ]
      pts2d, _, _, _, _ = cameras.project_point_radial( np.reshape(t3d, [-1, 3]), R, T, f, c, k, p )

      pts2d = np.reshape( pts2d, [-1, len(H36M_NAMES)*2] )
      sname = seqname[:-3]+"."+name+".h5" # e.g.: Waiting 1.58860488.h5
      t2d[ (subj, a, sname) ] = pts2d

  return t2d

def read_3d_data( camera_frame, rcams, superimp=False, changeorig=False,
  proc_gt=-1, lowpass=False ):
  """
  Loads 3d poses, zero-centres and normalizes them

  Args
   transform_world_to_camera(poses_set, rcams, f): camera_frame: boolean. Whether to convert the data to camera coordinates
    rcams: dictionary with camera parameters
  Returns
    train_set: dictionary with loaded 3d poses for training
    test_set: dictionary with loaded 3d poses for testing
    data_mean: vector with the mean of the 3d training data
    data_std: vector with the standard deviation of the 3d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
    train_root_positions: dictionary with the 3d positions of the root in train
    test_root_positions: dictionary with the 3d positions of the root in test
  """
  dim = 3 # reading 3d data
  print("\n[*] dimensions to use: ")
  print(DIMENSIONS_TO_USE)
  print()
  # Load 3d data
  data3d = load_data( dim, rcams, camera_frame, superimp, changeorig, proc_gt, lowpass )
  train_set = split_train_test( data3d, TRAIN_FILES, dim, camera_frame )
  test_set  = split_train_test( data3d, TEST_FILES, dim, camera_frame )
  
  # Compute normalization statistics
  complete_train = np.copy( np.vstack( list(train_set.values()) ).reshape((-1, DIMENSIONS*3)) )
  data_mean, data_std, dim_to_ignore, dim_to_use = \
    normalization_stats( complete_train, changeorig, dim )
  
  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, dim )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, dim )
  
  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def read_2d_predictions(changeorig):
  """
  Loads 2d data from precomputed Stacked Hourglass detections

  Args
    changeorig: change the origin of the system to ROOT_POSITION
  Returns
    train_set: dictionary with loaded 2d stacked hourglass detections for training
    test_set: dictionary with loaded 2d stacked hourglass detections for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """
  dim = 2 # reading 2d data
  data2d = load_data( dim )
  train_set = split_train_test( data2d, TRAIN_FILES, dim )
  test_set  = split_train_test( data2d, TEST_FILES, dim )
  
  # Compute normalization statistics
  complete_train = np.copy( np.vstack( list(train_set.values()) ).reshape((-1, DIMENSIONS*2)) )
  data_mean, data_std, dim_to_ignore, dim_to_use = \
    normalization_stats( complete_train, changeorig, dim )
  
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, dim )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, dim )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

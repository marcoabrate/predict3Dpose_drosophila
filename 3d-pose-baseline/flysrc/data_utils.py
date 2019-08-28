
"""Utility functions for dealing with Drosophila melanogaster data."""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pickle import load
import random

import cameras
import viz
import h5py
import glob
import copy
import sys
import signal_utils
import procrustes

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CAMERA_TO_USE = 1
CAMERA_PROJ = CAMERA_TO_USE

#FILES_CALIB = [os.path.join("flydata_calib/", f) \
#     for f in os.listdir("flydata_calib/") if os.path.isfile(os.path.join("flydata_calib/", f))]
#FILES_CALIB.sort()
TRAIN_FILES = [os.path.join("flydata_train/", f) \
     for f in os.listdir("flydata_train/") if os.path.isfile(os.path.join("flydata_train/", f))]
TRAIN_FILES.sort()
TEST_FILES = [os.path.join("flydata_test/", f) \
     for f in os.listdir("flydata_test/") if os.path.isfile(os.path.join("flydata_test/", f))]
TEST_FILES.sort()
'''
if CAMERA_TO_USE > 3:
  TRAIN_FILES_C = [f.replace("calib/calib", "train/pose_result")
     for f in FILES_CALIB if f.replace("calib/calib", "train/pose_result") in TRAIN_FILES]
  TEST_FILES_C = [f.replace("calib/calib", "test/pose_result") 
     for f in FILES_CALIB if f.replace("calib/calib", "test/pose_result") in TEST_FILES]
  TRAIN_FILES = TRAIN_FILES_C
  TEST_FILES = TEST_FILES_C
'''
FILES = TRAIN_FILES + TEST_FILES
FILE_NUM = len(FILES)
FILE_REF = FILES[FILE_NUM-1]

random.shuffle(TRAIN_FILES)
random.shuffle(TEST_FILES)

DIMENSIONS = 38
BODY_COXA = [0, 5, 10, 19, 24, 29]

ROOT_POSITIONS = []
if CAMERA_TO_USE < 4:
  LEG_TO_USE = "012"
  ROOT_POSITION = 0
  for i in range(3):
    if str(i) in LEG_TO_USE:
      ROOT_POSITIONS.append(ROOT_POSITION+i*5)
else:
  LEG_TO_USE = "543"
  ROOT_POSITION = 19
  for i in range(3):
    if str(i+3) in LEG_TO_USE:
      ROOT_POSITIONS.append(ROOT_POSITION+i*5)

LEGS = [list(range(5)), list(range(5,10)), list(range(10,15)),
  list(range(19,24)), list(range(24,29)), list(range(29,34))]
DIMENSIONS_TO_USE = []
for l in LEG_TO_USE:
  DIMENSIONS_TO_USE += LEGS[int(l)]
DIMENSIONS_TO_USE.sort()

def read_data(dir):
  with (open(dir, "rb")) as file:
    try:
      return load(file)
    except EOFError:
      return None

def load_data(dim, rcams=None, camera_frame=False, origin_bc=False, augment=False,
  procrustes=False, lowpass=False):
  """
  Loads data from disk, and puts it in an easy-to-acess dictionary
  Args
    dim: Integer={2,3}. Load 2 or 3-dimensional data
    rcams: dictionary (file, ncamera) with camera information
    camera_frame: project to camera coordinates
    origin_bc: origin in BODY COXAS
    augment: augment data adding two more projections to lateral cameras
    procrustes: apply procrustes analysis for all BODY COXA
    lowpass: lowpass data to smooth movements
  Returns
    dim = 3
      data: dictionary with keys k=(file, camera_proj)
        values: matrix (N, 38, 3) with 3d points data
    dim = 2
      data: dictionary with keys k=(file, camera_to_use)
        values: matrix (N, 38, 3) with 2d points data
  """

  dics = []
  dims = np.zeros((FILE_NUM+1), dtype=int)
  for idx, f in enumerate(FILES):
    print("[*] reading file {0}".format(f))
    if dim == 3:
      d = read_data(f)['points3d']
      dinit = d
      if camera_frame:
        d = transform_world_to_camera(dinit, rcams, CAMERA_PROJ, f)
        if f in TRAIN_FILES and augment:
          d = np.vstack((d, transform_world_to_camera(dinit, rcams, 0, f)))
          d = np.vstack((d, transform_world_to_camera(dinit, rcams, 2, f)))
      if origin_bc:
        d = origin_body_coxa_3d(d)
      dics.append(d)
      dims[idx+1] = dims[idx] + d.shape[0]
    else: # dim == 2
      d = read_data(f)['points2d'][CAMERA_TO_USE]
      if f in TRAIN_FILES and augment:
        d = np.vstack((d, read_data(f)['points2d'][0]))
        d = np.vstack((d, read_data(f)['points2d'][2]))
      if origin_bc:
        d = origin_body_coxa_2d(d)
      dics.append(d)
      dims[idx+1] = dims[idx] + d.shape[0]
  d_data = np.vstack(dics)
  print("[+] done reading data, shape: ", d_data.shape)
  
  if dim == 3 and lowpass:
    d_data = signal_utils.filter_batch(d_data) 

  data = {}
  for idx, f in enumerate(FILES):
    data[(f)] = d_data[dims[idx]:dims[idx+1]]
 
  if dim == 3 and procrustes:
    data = apply_procrustes(data)
 
  return data

def transform_world_to_camera(t3d_world, rcams, ncam, f):
  """
  Project 3d poses from world coordinate to camera coordinate system
  Args
    t3d_world: matrix Nx with 3d poses in world coordinates
    rcams: dictionary (file, ncamera) with camera information
    ncam: number of camera to project to
    f: file name
  Return
    t3d_camera: matrix with 3d poses in camera coordinate
  """
  t3d_camera = np.zeros(t3d_world.shape)
  R, T, _, _, _, intr = rcams[ (f, ncam) ]
  
  for i in range(t3d_world.shape[0]):
    camera_coord = cameras.world_to_camera_frame( t3d_world[i], R, T, intr )
    t3d_camera[i] = camera_coord
  return t3d_camera

def origin_body_coxa_3d(data3d):
  for i in range(data3d.shape[0]):
    for b in BODY_COXA:
      for coord in range(3):
        data3d[i,b:b+5,coord] -= data3d[i,b,coord]
  return data3d

def origin_body_coxa_2d(data2d):
  for i in range(data2d.shape[0]):
    ref = data2d[i,ROOT_POSITION]
    for b in BODY_COXA:
      if b != ROOT_POSITION:
        for coord in range(2):
          data2d[i,b:b+5,coord] += (ref[coord] - data2d[i,b,coord])
  return data2d

def separate_body_coxa_3d(list_of_data3d, rcams):
  dics = []
  for f in TEST_FILES:
    dinit = read_data(f)['points3d']
    d = transform_world_to_camera(dinit, rcams, CAMERA_PROJ, f)
    dics.append(d)
  data3d = np.vstack(dics)
  dists = []
  for bc in BODY_COXA:
    d = np.mean(data3d[:,bc] - data3d[:,BODY_COXA[0]], axis=0)
    for i in range(5):
      dists.append(d)
  dists = np.hstack(dists)
  for d in list_of_data3d:
    d[:,:15*3] += dists[:15*3]
    d[:,19*3:19*3+15*3] += dists[15*3:]

def separate_body_coxa_2d(data2d):
  dics = []
  for f in TEST_FILES:
    d = read_data(f)['points2d'][CAMERA_TO_USE]
    dics.append(d)
  d2d = np.vstack(dics)
  ref = np.mean(d2d[:,ROOT_POSITION], axis=0)
  for bc in BODY_COXA:
    if bc != ROOT_POSITION:
      dists = []
      for i in range(5):
        dists.append(ref - np.mean(d2d[:,bc], axis=0))
      dists = np.hstack(dists)
      data2d[:,bc*2:bc*2+5*2] -= dists
  return data2d

def apply_procrustes(data3d):
  """
    Superimpose data to one reference file FILE_REF
  """
  ground_truth = data3d[(FILE_REF)]
  for f in FILES:
    if f is FILE_REF:
      continue
    pts_t, d = procrustes.procrustes(data3d[(f)], ground_truth, False)
    data3d[(f)] = pts_t
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

def normalization_stats(complete_data, origin_bc, dim):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored
  Args
    complete_data: Nxd matrix with poses
    origin_bc: origin in BODY COXAS
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
  if origin_bc:
    dtu = [x for x in DIMENSIONS_TO_USE if x not in ROOT_POSITIONS]
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

def read_3d_data( camera_frame, rcams, origin_bc=False, augment=False,
  procrustes=False, lowpass=False ):
  """
  Loads 3d poses, zero-centres and normalizes them
  Args 
    camera_frame: project to camera coordinates
    rcams: dictionary (file, ncamera) with camera information
    origin_bc: origin in BODY COXAS
    augment: augment data adding two more projections to lateral cameras
    procrustes: apply procrustes analysis for all BODY COXA
    lowpass: lowpass data to smooth movements
  Returns
    train_set: dictionary with loaded 3d poses for training
    test_set: dictionary with loaded 3d poses for testing
    data_mean: vector with the mean of the 3d training data
    data_std: vector with the standard deviation of the 3d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """
  dim = 3 # reading 3d data
  print("\n[*] dimensions to use: ")
  print(DIMENSIONS_TO_USE)
  print()
  # Load 3d data
  data3d = load_data( dim, rcams, camera_frame, origin_bc, augment, procrustes, lowpass )
  train_set = split_train_test( data3d, TRAIN_FILES, dim, camera_frame )
  test_set  = split_train_test( data3d, TEST_FILES, dim, camera_frame )
  
  # Compute normalization statistics
  complete_train = np.copy( np.vstack( list(train_set.values()) ).reshape((-1, DIMENSIONS*3)) )
  data_mean, data_std, dim_to_ignore, dim_to_use = \
    normalization_stats( complete_train, origin_bc, dim )
  
  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, dim )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, dim )
  
  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def read_2d_predictions(origin_bc, augment):
  """
  Loads 2d data from precomputed Stacked Hourglass detections
  Args
    origin_bc: origin in BODY COXAS
    augment: augment data adding two more projections to lateral cameras
  Returns
    train_set: dictionary with loaded 2d stacked hourglass detections for training
    test_set: dictionary with loaded 2d stacked hourglass detections for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """
  dim = 2 # reading 2d data
  data2d = load_data( dim, origin_bc=origin_bc, augment=augment )
  train_set = split_train_test( data2d, TRAIN_FILES, dim )
  test_set  = split_train_test( data2d, TEST_FILES, dim )
  
  # Compute normalization statistics
  complete_train = np.copy( np.vstack( list(train_set.values()) ).reshape((-1, DIMENSIONS*2)) )
  data_mean, data_std, dim_to_ignore, dim_to_use = \
    normalization_stats( complete_train, origin_bc, dim )
  
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, dim )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, dim )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

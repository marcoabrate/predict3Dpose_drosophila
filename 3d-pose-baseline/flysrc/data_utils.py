
"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cameras
import viz
import h5py
import glob
import copy
import sys
import readpickle

TRAIN_SUBJECTS = range(0, 700)
TEST_SUBJECTS = range(700, 800)

DIMENSIONS = 38
DIMENSIONS_TO_USE = range(0, 15)

def load_data( bpath, subjects ):
  """
  Loads 2d ground truth from disk, and puts it in an easy-to-acess dictionary

  Args
    bpath: String. Pickle file where to load the data from
    subjects: List of integers. Subjects whose data will be loaded
    dim: Integer={2,3}. Load 2 or 3-dimensional data
  Returns:
    data: Dictionary with keys k=(subject, 0)
      values: matrix (38, 3) with the 3d points data
  """

  data = {}

  dic = readpickle.read_data(bpath)
  d3_data = dic['points3d']
  print('Reading subjects...')

  for subj in subjects:
    data[ (subj, 0) ] = d3_data[subj]
    
  return data


def load_stacked_hourglass(data_dir, subjects):
  """
  Load 2d detections from disk, and put it in an easy-to-acess dictionary.

  Args
    data_dir: string. Pickle file where to load the data from,
    subjects: list of integers. Subjects whose data will be loaded.
  Returns
    data: dictionary with keys k=(subject, camera)
    values: matrix (38, 2) with the 3d points data
  """

  data = {}
  
  dic = readpickle.read_data(data_dir)
  d2_data = dic['points2d']
  print('Reading subjects...')

  for subj in subjects:
    for c in range(7): # there are 7 cameras
      data[ (subj, c+1) ] = d2_data[c][subj]

  return data


def normalization_stats(complete_data, dim ):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    complete_data: nxd np array with poses
    dim. integer={2,3} dimensionality of the data
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

  # Encodes which 17 (or 14) 2d-3d pairs we are predicting
  dimensions_to_ignore = []
  if dim == 2:
    dimensions_to_use = np.sort( np.hstack((np.array(DIMENSIONS_TO_USE)*2,
      np.array(DIMENSIONS_TO_USE)*2+1)) )
    dimensions_to_ignore = np.delete(np.arange(DIMENSIONS*2), dimensions_to_use)
  else: # dim == 3
    dimensions_to_use = np.sort( np.hstack((np.array(DIMENSIONS_TO_USE)*3,
      np.array(DIMENSIONS_TO_USE)*3+1, np.array(DIMENSIONS_TO_USE)*3+2)) )
    dimensions_to_ignore = np.delete(np.arange(DIMENSIONS*3), dimensions_to_use)
  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def transform_world_to_camera(poses_set, cams, ncams=7 ):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras
    Return:
      t3d_camera: dictionary with keys (subject, camera)
        with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for subj in sorted( poses_set.keys() ):

      t3d_world = poses_set[ subj ]

      for c in range( ncams ):
        R, T, intr, d = cams[ c+1 ]
        camera_coord = cameras.world_to_camera_frame( t3d_world, R, T )
        #camera_coord = np.reshape( camera_coord, [-1, len(H36M_NAMES)*3] )

        t3d_camera[ (subj, c+1) ] = camera_coord
    
    return t3d_camera


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
    data[ key ] = np.reshape(data[ key ], (-1, dim*DIMENSIONS))
    data[ key ] = data[ key ][ :, dim_to_use ]
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out[ key ] = np.divide( (data[key] - mu), stddev )
  
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
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out[ key ] = np.multiply( data[key], stddev ) + mu

  return data_out

def unNormalize_batch(normalized_data, data_mean, data_std, dimensions_to_use):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing

  Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_use: list of dimensions that are used in the original data
  Returns
    orig_data: the input normalized_data, but unnormalized
  """
  T = normalized_data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality

  orig_data = np.zeros((T, D), dtype=np.float32)

  orig_data[:, dimensions_to_use] = normalized_data

  # Multiply times stdev and add the mean
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  orig_data = np.multiply(orig_data, stdMat) + meanMat
  return orig_data

def project_to_cameras( poses_set, cams, ncams=4 ):
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


def read_2d_predictions( data_dir ):
  """
  Loads 2d data from precomputed Stacked Hourglass detections

  Args
    data_dir: string. Pickle file where the data can be loaded from
  Returns
    train_set: dictionary with loaded 2d stacked hourglass detections for training
    test_set: dictionary with loaded 2d stacked hourglass detections for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """

  train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS )
  test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS )

  # Compute normalization statistics
  complete_train = np.copy( np.vstack( list(train_set.values()) ).reshape((-1, DIMENSIONS*2)) )
  data_mean, data_std, dim_to_ignore, dim_to_use = \
    normalization_stats( complete_train, dim=2 )
  
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, dim=2 )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, dim=2 )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def read_3d_data( data_dir, camera_frame, rcams ):
  """
  Loads 3d poses, zero-centres and normalizes them

  Args
    data_dir: string. Pickle file where the data can be loaded from
    camera_frame: boolean. Whether to convert the data to camera coordinates
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

  print("Dimensions to use: ")
  print(DIMENSIONS_TO_USE)
  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS )
  test_set  = load_data( data_dir, TEST_SUBJECTS )
  
  if camera_frame:
    train_set = transform_world_to_camera( train_set, rcams )
    test_set  = transform_world_to_camera( test_set, rcams )
    print("CAMERA_FRAME")

  # Apply 3d post-processing (centering around root)
  #train_set, train_root_positions = postprocess_3d( train_set )
  #test_set,  test_root_positions  = postprocess_3d( test_set )
  train_root_positions = np.array([0, 0, 0])
  test_root_positions = np.array([0, 0, 0])
  # Compute normalization statistics
  complete_train = np.copy( np.vstack( list(train_set.values()) ).reshape((-1, DIMENSIONS*3)) )
  data_mean, data_std, dim_to_ignore, dim_to_use = \
    normalization_stats( complete_train, dim=3 )
  
  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use, dim=3 )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use, dim=3 )
  
  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions


def postprocess_3d( poses_set ):
  """
  Center 3d points around root

  Args
    poses_set: dictionary with 3d data
  Returns
    poses_set: dictionary with 3d data centred around root (center hip) joint
    root_positions: dictionary with the original 3d position of each pose
  """
  root_positions = {}
  for k in poses_set.keys():
    # Keep track of the global position
    root_positions[k] = np.copy(poses_set[k][0])

    # Remove the root from the 3d position
    poses = poses_set[k]
    poses = poses - np.tile( poses[0], (poses.shape[0], 1) )
    poses_set[k] = poses

  return poses_set, root_positions

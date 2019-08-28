
"""Utilities to deal with the cameras of human3.6m"""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import data_utils
import viz

def world_to_camera_frame(P, R, T, intr):
  """
  Convert points from world to camera coordinates
  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    intr: 3x3 Camera intrinsic matrix
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3
  
  P = np.vstack(( P.T, np.ones((1,P.shape[0])) ))
  Rt = np.hstack((R, T))
  proj = Rt.dot(P)
  
  return proj.T

def load_camera_params( dic, ncamera ):
  """Load camera parameters
  Args
    dic: dictionary of data
    ncamera: number of the camera we are interested in
  Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    ce: 2x1 Camera center
    d: 6x1 Camera distortion coefficients
    intr: 3x3 Camera intrinsic matrix
  """
  
  if ncamera not in dic.keys():
    return None, None, None, None, None, None
   
  c = dic[ncamera]
  R = c['R']
  T = c['tvec'].reshape((-1,1))
  intr = c['intr']
  f = (intr[0,0]+intr[1,1])/2
  ce = intr[:2,2]
  if 'distort' in c.keys():
    d = c['distort']
  else:
    d = None
  
  return R, T, f, ce, d, intr

def load_cameras():
  """Loads the cameras
  Returns
    rcams: dictionary of 6 tuples per file containing the camera parameters
  """
  rcams = {}

  for f in data_utils.FILES:
    dic = data_utils.read_data(f)
    if data_utils.CAMERA_TO_USE < 4:
      for c in range(3):
        rcams[(f, c)] = load_camera_params( dic, c )
    else:
      for c in range(4,7):
        rcams[(f, c)] = load_camera_params( dic, c )

  return rcams

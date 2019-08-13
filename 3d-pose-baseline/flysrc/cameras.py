
"""Utilities to deal with the cameras of human3.6m"""

from __future__ import division

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import data_utils
import viz

def world_to_camera_frame(P, R, T, intr):
  """
  Convert points from world to camera coordinates

  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3
  
  #P = np.vstack(( P.T, np.ones((1,P.shape[0])) ))
  #Pnew = np.copy(P)*intr[0,0]
  #Rt = np.hstack((R, T))
  #proj = Rt.dot(P)
  #projnew = Rt.dot(Pnew)
  
  proj = R.dot(P.T - T)
  
  '''### tests ###
  realproj = intr.dot(proj)
  print(Rt)
  newintr = np.zeros(intr.shape)  
  newintr = intr/intr[0,0]
  realproj = newintr.dot(projnew)
  newintr[1,2] = 0
  fakeproj = newintr.dot(projnew)
  return realproj, fakeproj
  
  pixels = intr.dot(proj)
  pixels = pixels[:2,:] / pixels[2,:]
  '''
  return proj.T

def camera_to_world_frame(P, R, T):
  """Inverse of world_to_camera_frame

  Args
    P: Nx3 points in camera coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 points in world coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3
  
  X_cam = R.T.dot( P.T ) + T # rotate and translate

  return X_cam.T

def load_camera_params( dic, ncamera ):
  """Load h36m camera parameters

  Args
    dic: dictionary of data
    ncamera: number of the camera we are interested in
  Returns
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    ce: 2x1 Camera center
    d: 6x1 Camera distortion coefficients
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

  Args
    bpath: path to pickle file with camera data
  Returns
    rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
  """
  rcams = {}

  for f in data_utils.FILES:
    dic = data_utils.read_data(f)
    for c in range(3):
      rcams[(f, c)] = load_camera_params( dic, c )

  return rcams


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

def project_point_radial( P, R, T, f, c, k, p ):
  """
  Project points from 3d to 2d using camera parameters
  including radial and tangential distortion

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
  Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """

  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 2
  assert P.shape[1] == 3

  N = P.shape[0]
  X = R.dot( P.T - T ) # rotate and translate
  XX = X[:2,:] / X[2,:]
  r2 = XX[0,:]**2 + XX[1,:]**2

  radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) );
  tan = p[0]*XX[1,:] + p[1]*XX[0,:]

  XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )

  Proj = (f * XXX) + c
  Proj = Proj.T

  D = X[2,]

  return Proj, D, radial, tan, r2

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
  
  P = np.vstack(( P.T, np.ones((1,P.shape[0])) ))
  Pnew = np.copy(P)*intr[0,0]
  Rt = np.hstack((R, T))
  proj = Rt.dot(P)
  projnew = Rt.dot(Pnew)
  
  '''### tests ###
  realproj = intr.dot(proj)
  print(Rt)
  newintr = np.zeros(intr.shape)  
  newintr = intr/intr[0,0]
  realproj = newintr.dot(projnew)
  newintr[1,2] = 0
  fakeproj = newintr.dot(projnew)
  return realproj, fakeproj
  '''
  pixels = intr.dot(proj)
  pixels = pixels[:2,:] / pixels[2,:]
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
  
  c = dic[ncamera]
  if len(list(c.keys())) == 0:
    return None, None, None, None, None, None
  
  R = c['R']
  T = c['tvec'].reshape((-1,1))
  intr = c['intr']
  f = (intr[0,0]+intr[1,1])/2
  ce = intr[:2,2]
  d = c['distort']
  
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

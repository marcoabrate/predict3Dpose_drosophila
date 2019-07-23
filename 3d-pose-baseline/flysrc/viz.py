
"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import data_utils
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
import sys
import random

COLORS = ["#3498db", "#e74c3c", "#32cd32"]
TEST_COLORS = ["#000099", "#800000", "#006600"]

def show3Dpose(channels, ax, test_colors=False):
  """
  Visualize a 3d skeleton

  Args
    channels: the pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
  Returns
    Nothing. Draws on ax.
  """
  
  vals = channels[0].reshape( (-1, 3) )
  #  vals:
  #  x0  y0  z0
  #  x1  y1  z1
  #  x2  y2  z2
  #  ...
  #  x13  y13  z13
  I  = np.array([0,1,2,3,5,6,7,8,10,11,12,13]) # start points
  J  = np.array([1,2,3,4,6,7,8,9,11,12,13,14]) # end points
  cidx = 0

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, z, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    y *= -1
    z *= -1
    if not test_colors:
      ax.plot(x, y, z, lw=2, c=COLORS[cidx])
    else:
      ax.plot(x, y, z, lw=2, c=TEST_COLORS[cidx])
    if (i+1)%4 == 0:
      cidx += 1

  # NORMALIZE MEDIAN ???
  #def normalize_pose_3d(points3d, normalize_length=False, normalize_median=True):
  #  if normalize_median:
  #    points3d -= np.median(points3d.reshape(-1, 3), axis=0)
  
  # Get rid of the ticks and tick labels
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])

  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
  ax.set_zticklabels([])

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  #ax.w_xaxis.set_pane_color(white)
  #ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)

def show2Dpose(channels, ax):
  """
  Visualize a 2d skeleton

  Args
    channels: the pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
  Returns
    Nothing. Draws on ax.
  """

  vals = channels[0].reshape( (-1, 2) )
  #  vals:
  #  x0  y0
  #  x1  y1
  #  x2  y2
  #  ...
  #  x13  y13

  I  = np.array([0,1,2,3,5,6,7,8,10,11,12,13]) # start points
  J  = np.array([1,2,3,4,6,7,8,9,11,12,13,14]) # end points
  cidx = 0

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
    ax.plot(x, y, lw=2, c=COLORS[cidx])
    if (i+1)%4 == 0:
      cidx += 1

  # Get rid of the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Get rid of tick labels
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

  #RADIUS = 350 # space around the subject
  #xroot, yroot = vals[0,0], vals[0,1]
  #ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  #ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
  ax.set_xlim(0, 960)
  ax.set_ylim(0, 480)


  ax.set_aspect('equal')

def visualize_train_sample(train2d, train3d):
  # 1080p       = 1,920 x 1,080
  fig = plt.figure( figsize=(19.2, 10.8) )
  
  nrows = 3
  ncols = 2
  gs1 = gridspec.GridSpec(nrows, ncols*2)
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
  plt.axis('off')

  subplot_idx = 1
  camera = data_utils.CAMERA_TO_USE
  nsamples = nrows*ncols
  random_subjs = random.sample(list(train2d.keys()), nsamples)

  for (subj, _) in random_subjs:

    # Plot 2d pose
    ax1 = plt.subplot(gs1[subplot_idx-1])
    ax1.set_title("2D pose, subject {0}".format(subj), fontsize=5)
    p2d = train2d[ (subj, camera) ] 
    show2Dpose( p2d, ax1 )
    ax1.invert_yaxis()

    # Plot 3d pose
    ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
    ax2.set_title("3D pose, subject {0}".format(subj), fontsize=5)
    p3d = train3d[ (subj, 0) ]
    show3Dpose( p3d, ax2 )

    subplot_idx += 2

  plt.show()
  return random_subjs

def visualize_test_sample(test2d, test3d, predic):
  # 1080p       = 1,920 x 1,080
  fig = plt.figure( figsize=(19.2, 10.8) )
  
  nrows = 3
  ncols = 2
  gs1 = gridspec.GridSpec(nrows, ncols*3)
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
  plt.axis('off')

  subplot_idx = 1
  camera = data_utils.CAMERA_TO_USE
  nsamples = nrows*ncols
  random_subjs = np.random.randint(test2d.shape[0], size=(nsamples,))  

  for subj in random_subjs:
    # Plot 2d pose
    ax1 = plt.subplot(gs1[subplot_idx-1])
    ax1.set_title("2D pose, subject {0}".format(subj), fontsize=5)
    p2d = test2d[subj].reshape((1, -1))
    show2Dpose( p2d, ax1 )
    ax1.invert_yaxis()

    # Plot 3d ground truth
    ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
    ax2.set_title("Ground truth, subject {0}".format(subj), fontsize=5)
    p3d = test3d[subj].reshape((1, -1))
    show3Dpose( p3d, ax2 )

    # Plot 3d prediction
    ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
    ax3.set_title("Prediction, subject {0}".format(subj), fontsize=5)
    p3d = predic[subj].reshape((1, -1))
    show3Dpose( p3d, ax3, test_colors=True )

    subplot_idx += 3

  plt.show()
  return random_subjs

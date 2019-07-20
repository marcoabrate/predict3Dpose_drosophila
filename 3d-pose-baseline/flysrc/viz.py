
"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import data_utils
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
import sys

COLORS = ["#3498db", "#e74c3c", "#32cd32"]

def show3Dpose(channels, ax):
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
    ax.plot(x, y, z, lw=2, c=COLORS[cidx])
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

  gs1 = gridspec.GridSpec(3, 1*2) # 3 rows, 6 columns
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
  plt.axis('off')

  subplot_idx = 1
  camera = 1 # there are 7 cameras in total
  nsamples = 3
  for subj in np.random.randint(0, 700, size=(nsamples,)):

    # Plot 2d pose
    ax1 = plt.subplot(gs1[subplot_idx-1])
    p2d = train2d[ (subj, camera) ] 
    show2Dpose( p2d, ax1 )
    ax1.invert_yaxis()

    # Plot 3d pose
    ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
    p3d = train3d[ (subj, 0) ]
    show3Dpose( p3d, ax2 )

    subplot_idx += 2

  plt.show()

def visualize_test_sample(train2d, train3d, poses3d):
  # 1080p       = 1,920 x 1,080
  fig = plt.figure( figsize=(19.2, 10.8) )

  gs1 = gridspec.GridSpec(5, 9) # 5 rows, 9 columns
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
  plt.axis('off')

  subplot_idx = 1
  camera = 1 # there are 7 cameras in total
  nsamples = 15
  for subj in np.random.randint(0, 700, size=(nsamples,)):

    # Plot 2d pose
    ax1 = plt.subplot(gs1[subplot_idx-1])
    p2d = train2d[ (subj, camera) ]
    show2Dpose( p2d, ax1 )
    ax1.invert_yaxis()

    # Plot 3d pose
    ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
    p3d = train3d[ (subj, 0) ]
    show3Dpose( p3d, ax2 )

    subplot_idx += 2

  plt.show()

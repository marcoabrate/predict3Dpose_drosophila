
"""Functions to visualize human poses"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py
import os
import sys
import random

import data_utils
import signal_utils

if data_utils.CAMERA_TO_USE < 4:
  I  = np.array([0,1,2,3,5,6,7,8,10,11,12,13]) # start points
  J  = np.array([1,2,3,4,6,7,8,9,11,12,13,14]) # end points
else:
  I  = np.array([19,20,21,22,24,25,26,27,29,30,31,32]) # start points
  J  = np.array([20,21,22,23,25,26,27,28,30,31,32,33]) # end points

COLORS = ["#3498db", "#e74c3c", "#32cd32"]
TEST_COLORS = ["#000099", "#800000", "#006600"]

def show3Dpose(channels, ax, test_colors=False):
  """
  Visualize a 3d skeleton
  Args
    channels: the pose to plot.
    ax: matplotlib 3d axis to draw on
    test_colors: use the color for test images
  """
  
  vals = channels[0].reshape( (-1, 3) )
  #  vals:
  #  x0  y0  z0
  #  etc ...
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
  """

  vals = channels[0].reshape( (-1, 2) )
  #  vals:
  #  x0  y0
  #  etc ...
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

  ax.set_xlim(0, 960)
  ax.set_ylim(0, 480)

  ax.set_aspect('equal')

def visualize_train_sample(train2d, train3d, camera_frame):
  # 1080p       = 1,920 x 1,080
  fig = plt.figure( figsize=(19.2, 10.8) )
  
  nrows = 3
  ncols = 2
  gs1 = gridspec.GridSpec(nrows, ncols*2)
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes
  plt.axis('off')

  subplot_idx = 1
  camera = data_utils.CAMERA_TO_USE
  projcamera = data_utils.CAMERA_PROJ
  nsamples = nrows*ncols
  (random_file, _) = random.choice( list(train2d.keys()) )
  random_subjs = random.sample(range(train2d[ (random_file, camera) ].shape[0]), nsamples)

  for subj in random_subjs:

    # Plot 2d pose
    ax1 = plt.subplot(gs1[subplot_idx-1])
    ax1.set_title("2D pose, subject {0}".format(subj), fontsize=5)
    p2d = train2d[ (random_file, camera) ][subj].reshape((1,-1))
    show2Dpose( p2d, ax1 )
    ax1.invert_yaxis()
    
    if not camera_frame:
      # Plot 3d pose
      ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
      ax2.set_title("3D pose, subject {0}".format(subj), fontsize=5)
      p3d = train3d[ (random_file, 0) ][subj].reshape((1,-1))
      show3Dpose( p3d, ax2 ) 
    else:
      # Plot projected 3d pose
      ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
      ax2.set_title("projected 3D pose, subject {0}".format(subj), fontsize=5)
      p3d = train3d[ (random_file, projcamera) ][subj].reshape((1,-1))
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

def update_files_graph(num, files_dim, sets3d, ax3d):
  ax3d.cla()
  ax3d.set_title("3D pose {0}".format(num))
  channels = []
  for d in files_dim:
    channels.append(sets3d[d+num].reshape((1,-1)))
  get_3d_pose(channels, ax3d)

def visualize_files_animation(train3d, test3d):
  """
  Visualize animation of all files on the same plot
  Args
    train3d: dictionary with 3d data used for training
    test3d: dictionary with 3d data used fot testing
  """
  fig = plt.figure(figsize=(19.2, 10.8))
  files_dim = np.zeros((data_utils.FILE_NUM+1), dtype=int)
  values = []
  idx = 0
  for v in train3d.values():
    files_dim[idx+1] = files_dim[idx] + v.shape[0]
    values.append(v)
    idx += 1
  for v in test3d.values():
    files_dim[idx+1] = files_dim[idx] + v.shape[0]
    values.append(v)
    idx += 1
  files_dim = files_dim[:-1]
  sets3d = np.copy( np.vstack( values ) )
  ax3d = fig.add_subplot(111, projection='3d')
  
  ax3d.set_title("3D pose")
  channels = []
  for d in files_dim:
    channels.append(sets3d[d].reshape((1,-1)))
  get_3d_pose(channels, ax3d)

  ani = animation.FuncAnimation(fig, update_files_graph, min(files_dim[1:])-1,
    fargs=(files_dim, sets3d, ax3d), interval=10, blit=False)
  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=15, bitrate=1800)
  #ani.save('predictions.mp4', writer=writer)
  plt.show()

def update_files_oneatatime(num, data3d, files, ax3d):
  f = files[int(num/100)]
  ax3d.cla()
  title = f[7:]
  if f in data_utils.FILES_CALIB:
    title += "\nCalib"
  ax3d.set_title(title)
  ax3d.set_xlim([-2, 2])
  ax3d.set_ylim([-2, 0])
  ax3d.set_zlim([-1, 0.5])
  channels = [ data3d[num] ]
  get_3d_pose(channels, ax3d)

def visualize_files_oneatatime(train3d, test3d):
  """
  Visualize animation of all files one after the other
  Args
    train3d: dictionary with 3d data used for training
    test3d: dictionary with 3d data used fot testing
  """
  data3d = []
  files = []
  for k in train3d.keys():
    f, _ = k
    data3d.append(train3d[k][:100])
    files.append(f)
  for k in test3d.keys():
    f, _ = k
    data3d.append(test3d[k][:100])
    files.append(f)
  data3d = np.vstack(data3d)

  fig = plt.figure(figsize=(19.2, 10.8))
  ax3d = fig.add_subplot(111, projection='3d')
  ax3d.set_title(files[0])
  channels = [ data3d[0] ]
  get_3d_pose(channels, ax3d)
  ani = animation.FuncAnimation(fig, update_files_oneatatime, len(files)*100,
    fargs=(data3d, files, ax3d), interval=10, blit=False)
  plt.show()

def update_test_graph(num, test3d, predic, ax3d):
  ax3d.cla()
  ax3d.set_title("prediction in RED {0}".format(num))
  channels = [test3d[num].reshape((1,-1)), predic[num].reshape((1,-1))]
  ax3d.set_xlim([-2, 2])
  ax3d.set_ylim([-2, 0])
  ax3d.set_zlim([-1, 0.5])
  get_3d_pose(channels, ax3d)

def visualize_test_animation(test3d, predic):
  """
  Visualize animation of the test and predicted data, on the same plot
  Args
    test3d: 3d data used fot testing
    predic: predicted 3d data
  """
  test3d = signal_utils.filter_batch(test3d.reshape((-1,data_utils.DIMENSIONS,3)))
  predic = signal_utils.filter_batch(predic.reshape((-1,data_utils.DIMENSIONS,3)))
  fig = plt.figure(figsize=(19.2, 10.8))
  ax3d = fig.add_subplot(111, projection='3d')
  ax3d.set_title("prediction in RED 0")
  channels = [test3d[0].reshape((1,-1)), predic[0].reshape((1,-1))]
  get_3d_pose(channels, ax3d)

  ani = animation.FuncAnimation(fig, update_test_graph, test3d.shape[0],
    fargs=(test3d, predic, ax3d), interval=10, blit=False)
  Writer = animation.writers['ffmpeg']
  writer = Writer(fps=15, bitrate=1800)
  #ani.save('predictions.mp4', writer=writer)
  plt.show()

def get_3d_pose(channels, ax):
  cidx = 0
  colors = ["#004100", "#009600", "#00ff00", "#410000", "#960000", "#ff0000",\
    "#000041", "#000096", "#0000ff"]
  for ch in channels:
    vals = ch.reshape((-1, 3))
    for i in np.arange(len(I)):
      x, z, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
      y *= -1
      z *= -1
      ax.plot(x, y, z, lw=2, c=colors[cidx])
      if (i+1)%4 == 0:
        cidx+=1
      if cidx > 8:
        cidx = 0

def get_2d_pose(channel, ax):
  vals = channel.reshape((-1, 2))  
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
  ax.set_xlim(0, 960)
  ax.set_ylim(0, 480)

  ax.set_aspect('equal')
  ax.invert_yaxis()

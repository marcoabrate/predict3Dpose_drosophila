import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import h5py
import os
import sys
import random

I  = np.array([0,1,2,3,5,6,7,8,10,11,12,13]) # start points
J  = np.array([1,2,3,4,6,7,8,9,11,12,13,14]) # end points
I_r  = np.array([19,20,21,22,24,25,26,27,29,30,31,32]) # start points
J_r  = np.array([20,21,22,23,25,26,27,28,30,31,32,33]) # end points

def get_3d_pose(channels, ax):
  cidx = 0
  colors = ["#004100", "#009600", "#00ff00", "#410000", "#960000", "#ff0000"]
  for ch in channels:
    vals = ch.reshape((-1, 3))
    for i in np.arange(len(I)):
      x, z, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
      y *= -1
      z *= -1
      ax.plot(x, y, z, lw=2, c=colors[cidx])
      if (i+1)%4 == 0:
        cidx+=1
      if cidx > 5:
        cidx = 0

def update_test_graph(num, test3d_l, predic_l, test3d_r, predic_r, ax3d):
  ax3d.cla()
  ax3d.set_title("prediction in RED {0}".format(num))
  channels = [test3d_l[num].reshape((1,-1)), predic_l[num].reshape((1,-1)),
    test3d_r[num,I_r[0]*3:].reshape((1,-1)), predic_r[num,I_r[0]*3:].reshape((1,-1))]
  get_3d_pose(channels, ax3d)

LEFT_DIR = "tr_all_te3-24_oldcam_200epochs_origBC_camproj_size1024_dropout0.5_0.001"
RIGHT_DIR = "tr_all_te0-8_5epochs_origBC_size1024_dropout0.5_0.001"
test3d_l = np.load(LEFT_DIR+"/test_poses3d.npy")
predic_l = np.load(LEFT_DIR+"/test_decout.npy")
test3d_r = np.load(RIGHT_DIR+"/test_poses3d.npy")
predic_r = np.load(RIGHT_DIR+"/test_decout.npy")

fig = plt.figure(figsize=(19.2, 10.8))
ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_title("prediction in RED 0")
channels = [test3d_l[0].reshape((1,-1)), predic_l[0].reshape((1,-1)),
  test3d_r[0,I_r[0]*3:].reshape((1,-1)), predic_r[0,I_r[0]*3:].reshape((1,-1))]
get_3d_pose(channels, ax3d)

ani = animation.FuncAnimation(fig, update_test_graph, test3d_l.shape[0],
  fargs=(test3d_l, predic_l, test3d_r, predic_r, ax3d), interval=10, blit=False)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, bitrate=1800)
#ani.save('predictions.mp4', writer=writer)
plt.show()


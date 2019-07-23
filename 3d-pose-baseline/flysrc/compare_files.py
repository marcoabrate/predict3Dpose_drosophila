import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import data_utils
import viz
import sys

def get_3d_pose(channels, ax):
  vals = channels.reshape((-1, 3))
  I  = np.array([0,1,2,3,5,6,7,8,10,11,12,13]) # start points
  J  = np.array([1,2,3,4,6,7,8,9,11,12,13,14]) # end points
  cidx = 0
  for i in np.arange(len(I)):
    x, z, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    y *= -1
    z *= -1
    ax.plot(x, y, z, lw=2)
  
def update_graph(num, idata3d, ax):
  plt.cla()
  #ax.set_xlim3d([-0.6, 1.1])
  #ax.set_ylim3d([0.0, 1.0])
  ax.set_title("3D Test, time={0}".format(num))
  get_3d_pose(idata3d[num].reshape((-1,1)), ax)
  
DIMENSIONS = 38

data_dir = "flydata/"
file_dim = 899
file_num = 3

subjects = list(range(file_dim*file_num+1))
data2d = data_utils.load_data(data_dir, subjects, dim=2)
data3d = data_utils.load_data(data_dir, subjects, dim=3)
data2d = np.copy(np.vstack(list(data2d.values())).reshape((-1, DIMENSIONS*2)))
data3d = np.copy(np.vstack(list(data3d.values())).reshape((-1, DIMENSIONS*3)))
print("\n[+] done reading data")
print(data2d.shape, data3d.shape)
print("\n")

for i in range(1, file_num+1):
  print("\n[*] visualizing from file {0}".format(i))
  idata2d = np.copy(data2d[(i-1)*file_dim:i*file_dim])
  idata3d = np.copy(data3d[(i-1)*file_dim:i*file_dim])
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  title = ax.set_title('3D Test')
  ax.set_xlim3d([-0.6, 1.1])
  ax.set_ylim3d([0.0, 1.0])
  get_3d_pose(idata3d[0].reshape((-1,1)), ax) 
  
  ani = animation.FuncAnimation(fig, update_graph, 899, fargs=(idata3d, ax), interval=10, blit=False)
  plt.show()

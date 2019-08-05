import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import data_utils
import viz
import procrustes
import sys

data_dir = "flydata/"
FILE_DIM = 899
FILE_NUM = 3
SUPERIMP_POSITION = 0
ROOT_POSITION = 0
BODY_COXA = [0, 5, 10, 19, 24, 29]

def apply_procrustes(data3d, gt):
  ground_truth = data3d[FILE_DIM*gt:FILE_DIM*(gt+1)]
  for i in range(FILE_NUM):
    if i == gt:
      continue
    pts_t, d = procrustes.procrustes(data3d[FILE_DIM*i:FILE_DIM*(i+1)], ground_truth, False)
    data3d[FILE_DIM*i:FILE_DIM*(i+1)] = pts_t
  return data3d

def superimpose(data3d):
  for coord in range(3):
    dist = np.mean(data3d[:FILE_DIM,SUPERIMP_POSITION,coord]) -\
       np.mean(data3d[FILE_DIM:,SUPERIMP_POSITION,coord])
    data3d[FILE_DIM:,:,coord] += dist
  return data3d

def change_origin(data3d):
  for coord in range(3):
    origin = np.mean(data3d[:,ROOT_POSITION,coord])
    data3d[:,:,coord] -= origin
    data3d[:,ROOT_POSITION,coord] = 0
  return data3d

def print_data_info(data):
  xyz = [] 
  for i in range(data.shape[2]):
    xyz.append(data[:,:,i])
  for i in range(len(xyz)):
    print("\tX{0}".format(i))
    print("\tmean: ", np.mean(xyz[i].flatten()))
    print("\tmin: ", np.min(xyz[i].flatten()))
    print("\tmax: ", np.max(xyz[i].flatten()))
    print("\tinterval: ", np.abs(np.max(xyz[i].flatten())-np.min(xyz[i].flatten())))
  return

def get_3d_pose(channels, ax):
  I  = np.array([0,1,2,3,5,6,7,8,10,11,12,13]) # start points
  J  = np.array([1,2,3,4,6,7,8,9,11,12,13,14]) # end points
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
  
def update_graph(num, data3d, ax):
  plt.cla()
#  ax.set_xlim3d([-2, 2])
#  ax.set_ylim3d([-2, 2])
#  ax.set_zlim3d([-2, 0])
  ax.set_title("3D Test, time={0}".format(num))
  channels = [data3d[num].reshape((-1,1)), data3d[FILE_DIM+num].reshape((-1,1)),\
    data3d[2*FILE_DIM+num].reshape((-1,1))]
  get_3d_pose(channels, ax)
  
subjects = list(range(FILE_DIM*FILE_NUM+1))
data2d = data_utils.load_data(data_dir, subjects, dim=2)
data3d = data_utils.load_data(data_dir, subjects, dim=3)
data2d = np.copy(np.vstack(list(data2d.values())).reshape((-1, data_utils.DIMENSIONS, 2)))
data3d = np.copy(np.vstack(list(data3d.values())).reshape((-1, data_utils.DIMENSIONS, 3)))

data3d = apply_procrustes(data3d, 1)
#data3d = change_origin(data3d)

print("\n[+] done reading data")
print(data2d.shape, data3d.shape)

pdata3d = np.copy(data3d.reshape((-1, data_utils.DIMENSIONS*3)))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
#ax.set_xlim3d([-3.5, 3.5])
#ax.set_ylim3d([-1.8, 3.6])
#ax.set_zlim3d([0.2, 2.5])
channels = [pdata3d[0].reshape((-1,1)), pdata3d[FILE_DIM].reshape((-1,1)),\
   pdata3d[2*FILE_DIM].reshape((-1,1))]
get_3d_pose(channels, ax) 

ani = animation.FuncAnimation(
  fig, update_graph, FILE_DIM, fargs=(pdata3d, ax), interval=10, blit=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, bitrate=1800)
#ani.save('compare.mp4', writer=writer)
plt.show()

for i in range(1, FILE_NUM+1):
  print("\n[*] visualizing from file {0}".format(i))  
  idata2d = np.copy(data2d[(i-1)*FILE_DIM:i*FILE_DIM])
  idata3d = np.copy(data3d[(i-1)*FILE_DIM:i*FILE_DIM])
  #print("2D data information:")
  #print_data_info(idata2d)
  print("\n3D data information:")
  print_data_info(idata3d)

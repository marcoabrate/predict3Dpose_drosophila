import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, cycler

train_dir = "test_2epochs_origBC_camproj_size1024_dropout0.5_0.001/"
with open(train_dir+"losses.pkl", 'rb') as f:
  losses = pickle.load(f)
with open(train_dir+"errors.pkl", 'rb') as f:
  errors = pickle.load(f)
with open(train_dir+"joint_errors.pkl", 'rb') as f:
  joint_errors = pickle.load(f)

plt.plot(losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

xerrors, yerrors, zerrors, dists = [],[],[], []
for e in errors:
  xerrors.append(e[0])
  yerrors.append(e[1])
  zerrors.append(e[2])
  dists.append(np.sqrt(e[0]**2+e[1]**2+e[2]**2))
plt.plot(xerrors, marker="X")
plt.plot(yerrors, marker="v")
plt.plot(zerrors, marker=">")
plt.plot(dists, marker="o")
plt.legend(["X error", "Y error", "Z error", "L2 error"])
plt.xlabel("Epoch")
plt.ylabel("Distance (mm)")
plt.show()

joints = [[] for i in range(12)]
for je in joint_errors:
  for i in range(12):
    joints[i].append(je[i])
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0.6, 1, 4)))
for leg in range(3):
  for i in range(leg*4, leg*4+4):
    plt.plot(joints[i])
  legend = []
  for i in range(4):
    legend.append("Joint %d"%(i+1))
  plt.legend(legend)
  plt.title("Leg %d"%(leg+1))
  plt.xlabel("Epoch")
  plt.ylabel("L2 error")
  plt.show()

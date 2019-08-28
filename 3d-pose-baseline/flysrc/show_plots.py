import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, cycler

joints_name = ["Coxa-femur", "Femur-tibia", "Tibia-tarsus", "Tarsus tip"]

train_dir = "final_model_200epochs_origBC_camproj_size1024_dropout0.5_0.001/"
with open(train_dir+"losses.pkl", 'rb') as f:
  losses = pickle.load(f)
with open(train_dir+"errors.pkl", 'rb') as f:
  errors = pickle.load(f)
with open(train_dir+"joint_errors.pkl", 'rb') as f:
  joint_errors = pickle.load(f)

plt.plot(losses)
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Loss (mm)", fontsize=13)
plt.minorticks_on()
plt.grid(True, which='major', axis='y')
plt.grid(True, which='minor', axis='y', linestyle='--')
plt.savefig("loss.png")
plt.show()

xerrors, yerrors, zerrors, dists = [],[],[], []
for e in errors:
  xerrors.append(e[0])
  yerrors.append(e[1])
  zerrors.append(e[2])
  dists.append(np.sqrt(e[0]**2+e[1]**2+e[2]**2))
plt.plot(xerrors, marker="X", markersize=1.5, color="#28de37")
plt.plot(yerrors, marker="v", markersize=1.5, color="#26e0d4")
plt.plot(zerrors, marker=">", markersize=1.5, color="#2651e0")
plt.legend(["X", "Y", "Z"], prop={'size': 9})
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("Error (mm)", fontsize=13)
plt.minorticks_on()
plt.grid(True, which='major', axis='y')
plt.grid(True, which='minor', axis='y', linestyle='--')
plt.savefig("error_coordwise.png")
plt.show()

joints = [[] for i in range(12)]
for je in joint_errors:
  for i in range(12):
    joints[i].append(je[i])
cmap = plt.cm.coolwarm
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0.7, 1, 4)))
for leg in range(3):
  for i in range(leg*4, leg*4+4):
    plt.plot(joints[i])
  legend = []
  for i in range(4):
    legend.append(joints_name[i])
  if leg > 1:
    plt.legend(legend, prop={'size': 9}, loc='center right')
  else:
    plt.legend(legend, prop={'size': 9}, loc='upper right')
  plt.xlabel("Epoch", fontsize=13)
  plt.ylabel("L2 error (mm)", fontsize=13)
  plt.minorticks_on()
  plt.grid(True, which='major', axis='y')
  plt.grid(True, which='minor', axis='y', linestyle='--')
  title = "limb"+str(leg+1)+".png"
  plt.savefig(title)
  plt.show()

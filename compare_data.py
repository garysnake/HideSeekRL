import numpy as np
from matplotlib import pyplot as plt
import sys

if len(sys.argv) < 4:
    print("*** ERROR: not enough arguments ***")
    print("How to run: python3 compare_data.py file1,file2,file3 name1,name2,name3 save_name")
    print()
    exit(-1)

def read_g(filename):
    with open(filename, "r") as f:
        gs = [float(g) for g in f.read().splitlines()[0].split(",")]
    return np.asarray(gs)

filenames = sys.argv[1].split(",")
gs = [read_g(filename) for filename in filenames]
episodes = [np.arange(len(g)) for g in gs]
labels = sys.argv[2].split(",")

fig, ax = plt.subplots()
for i in range(len(gs)):
    ax.plot(episodes[i], gs[i], label=labels[i])
plt.title("Hider G_0 comparison")
ax.set_xlabel('episode')
ax.set_ylabel('G_0')
ax.legend()
plt.savefig(sys.argv[3], dpi=300)
plt.clf()
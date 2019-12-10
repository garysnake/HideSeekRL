import numpy as np
from gridworld import GridWorld_hybrid_state, GridWorld_coord, GridWorld_diag
from train import random_baseline, REINFORCE, SarsaLambda, actor_critic
from ppo import train_ppo

from matplotlib import pyplot as plt
import sys

algorithm_list = ["random", "reinforce", "actor_critic", "ppo"]
algorithm_dict = {}
algorithm_dict["random"] = random_baseline
algorithm_dict["reinforce"] = REINFORCE
algorithm_dict["actor_critic"] = actor_critic
algorithm_dict["ppo"] = train_ppo

wall_list = ["none", "two_walls", "cross"]

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print()
        print("*** ERROR: not enough arguments ***")
        print("Usage: python3 test.py num_hiders num_seekers algorithm_name num_epochs num_episodes episode_length")
        print("Available algorithms:", algorithm_list)
        print("Available wall types:", wall_list)
        print()
        exit(-1)
    # VARIABLES
    num_epochs = int(sys.argv[4]) # number of epochs
    num_episodes = int(sys.argv[5]) # number of episodes per epoch
    episode_length = int(sys.argv[6])
    num_x = 12
    num_y = 12
    num_hiders = int(sys.argv[1])
    num_seekers = int(sys.argv[2])
    hide_vis = 3
    seek_vis = 3
    algorithm_to_test = algorithm_dict[sys.argv[3]]
    wall_type = sys.argv[7]
    file_name = sys.argv[3] +"_" + wall_type + "_%d-%d" % (num_hiders, num_seekers)


    grid_world = GridWorld_diag(num_x, num_y, num_hiders, num_seekers, hide_vis, seek_vis, episode_length, wall_type)

    # Main testing framework
    hide_gs = []
    seek_gs = []
    for i in range(num_epochs):
        print("Epoch", i + 1)
        h_g, s_g = algorithm_to_test(grid_world, num_episodes)
        hide_gs.append(h_g)
        seek_gs.append(s_g)
    hide_gs = np.mean(hide_gs, axis=0)
    seek_gs = np.mean(seek_gs, axis=0)
    # Save gs
    with open("data/" + file_name + ".txt", "w") as f:
        f.write(",".join([repr(g) for g in hide_gs]))
        f.write("\n")
        f.write(",".join([repr(g) for g in seek_gs]))
    # Plot experiment result
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(hide_gs)), hide_gs, label="hide g_0s")
    ax.plot(np.arange(len(seek_gs)), seek_gs, label="seek g_0s")
    plt.title("%s, Num epochs=%d" % (sys.argv[3], num_epochs))
    ax.set_xlabel('episode')
    ax.set_ylabel('G_0')
    ax.legend()
    plt.savefig("img/" + file_name + ".png", dpi=300)
    plt.clf()


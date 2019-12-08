import numpy as np
from gridworld import GridWorld_hybrid_state, GridWorld_coord
from train import random_baseline, REINFORCE
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # VARIABLES
    num_epochs = 5 # number of epochs
    N_EPISODES = 100 # number of episodes per epoch
    episode_lenth = 1000
    num_x = 10
    num_y = 10
    num_hiders = 1
    num_seekers = 2
    hide_vis = 3
    seek_vis = 3

    grid_world = GridWorld_coord(num_x, num_y, num_hiders, num_seekers, hide_vis, seek_vis, episode_lenth)

    

    # Change below variables to test different algorithms    
    # algorithm_to_test = random_baseline
    algorithm_to_test = random_baseline
    algorithm_name = "Random policy with half episode reward"

    # Main testing framework
    hide_gs = []
    seek_gs = []
    for i in range(num_epochs):
        print("Epoch", i + 1)
        h_g, s_g = algorithm_to_test(grid_world, N_EPISODES)
        hide_gs.append(h_g)
        seek_gs.append(s_g)
    hide_gs = np.mean(hide_gs, axis=0)
    seek_gs = np.mean(seek_gs, axis=0)
    # Plot experiment result
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(hide_gs)), hide_gs, label="hide g_0s")
    ax.plot(np.arange(len(seek_gs)), seek_gs, label="seek g_0s")
    plt.title("%s, Num epochs=%d, Episode length=%d" % (algorithm_name, num_epochs, episode_lenth))
    ax.set_xlabel('episode')
    ax.set_ylabel('G_0')
    ax.legend()
    plt.savefig('img/%s.png' % (algorithm_name,), dpi=300)
    plt.clf()


import numpy as np
from gridworld import GridWorld_hybrid_state, GridWorld_coord, GridWorld_diag
from train import random_baseline, REINFORCE, SarsaLambda, actor_critic
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # VARIABLES
    num_epochs = 1 # number of epochs
    N_EPISODES = 750 # number of episodes per epoch
    episode_length = 200
    num_x = 12
    num_y = 12
    num_hiders = 1
    num_seekers = 1
    hide_vis = 3
    seek_vis = 3

    grid_world = GridWorld_diag(num_x, num_y, num_hiders, num_seekers, hide_vis, seek_vis, episode_length)

    # Change below variables to test different algorithms    
    # algorithm_to_test = random_baseline, REINFORCE, SarsaLambda, actor_critic
    algorithm_to_test = actor_critic
    algorithm_name = "actor-critic_2_walls" + "_%d-%d" % (num_hiders, num_seekers)

    # algorithm_name = "actor-critic_"

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
    # Temporarily save gs
    # with open("data.txt", "w") as f:
    #     f.write(",".join([repr(g) for g in hide_gs]))
    #     f.write("\n")
    #     f.write(",".join([repr(g) for g in seek_gs]))
    # Plot experiment result
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(hide_gs)), hide_gs, label="hide g_0s")
    ax.plot(np.arange(len(seek_gs)), seek_gs, label="seek g_0s")
    plt.title("%s, Num epochs=%d" % (algorithm_name, num_epochs))
    ax.set_xlabel('episode')
    ax.set_ylabel('G_0')
    ax.legend()
    algorithm_name = "_".join(algorithm_name.split())
    plt.savefig('img/%s.png' % (algorithm_name,), dpi=300)
    plt.clf()


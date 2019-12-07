import numpy as np
from gridworld import GridWorld

class RandomPolicy(object):
    def __init__(self, nA, p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self, state, action=None):
        return self.p[action]

    def action(self, state):
        return np.random.choice(len(self.p), p=self.p)

if __name__ == "__main__":
    # VARIABLES
    N_EPISODES = 1
    episode_lenth = 5
    num_x = 10
    num_y = 10
    num_hiders = 1
    num_seekers = 2
    hide_vis = 3
    seek_vis = 3

    grid_world = GridWorld(num_x, num_y, num_hiders, num_seekers, hide_vis, seek_vis, episode_lenth)
    behavior_policy = RandomPolicy(5)
    # Initial world
    print("********************* START TRAINING *********************")
    for e in range(N_EPISODES):
        init_h, init_s = grid_world.reset()
        hider_states = [init_h]
        seeker_states = [init_s]
        while True:
            grid_world.print_world()
            h_action = behavior_policy.action(None)
            s_action = behavior_policy.action(None)
            print("Hider trying", h_action, "Seeker trying", s_action)
            hs, ss, hr, sr, done = grid_world.step([h_action] * num_hiders, [s_action] * num_seekers)
            hider_states.append(hs)
            seeker_states.append(ss)
            print("Hider r", hr, "Seeker r", sr)
            if done:
                break

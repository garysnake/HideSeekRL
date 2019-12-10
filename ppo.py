import numpy as np
import torch
import torch.nn as nn


class PPO(nn.module):
    def __init__(self, state_dim_list, action_dim, hidden_layer, lr, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.state_dim_list = self.state_dim_list
        state_dim = np.prod(state_dim_list)

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_layer),
            nn.Tanh(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.Tanh(),
            nn.Linear(hidden_layer, action_dim),
            nn.Softmax(dim=-1)
            )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.loss = nn.MSELoss()

    def update(self, memory):


def train_ppo(grid_world, num_episodes, gamma=.9):
    lr = 0.002
    K_epochs = 4
    eps_clips = 0.2

    state_dim_list = grid_world.state_dim_list
    num_actions = grid_world.env_spec.nA

    ppo = PPO(state_dim_list, num_actions, hidden_layer, lr, gamma, K_epochs, eps_clip)

    num_hiders = grid_world.numHider
    num_seekers = grid_world.numSeeker
    G0_hide = []
    G0_seek = []

    for episode in range(num_episodes):
        gamma_t = 1
        if episode % 10 == 0:
            print("\tEpisode", episode)
        old_h_state, old_s_state = grid_world.reset()
        R_h = [np.NINF] # Padding for index adjustment
        R_s = [np.NINF] # Padding for index adjustment

        # Generate an episode
        while True:
            h_actions = []
            for h in range(num_hiders):
                h_actions.append(actor_hide(old_h_state[h]))
            s_actions = []
            for s in range(num_seekers):
                s_actions.append(actor_seek(old_s_state[s]))
            ### DEBUG ###
            if episode % 50 == 0:
                grid_world.save_world("data/ac_big_walls.txt", episode)
            h_state, s_state, reward_h, reward_s, done = grid_world.step(h_actions, s_actions)
            R_h.append(reward_h)
            R_s.append(reward_s)
            for h in range(num_hiders):
                delta_h = reward_h + gamma * critic_hide(h_state[h]) - critic_hide(old_h_state[h])
                critic_hide.update(old_h_state[h], delta_h)
                actor_hide.update(old_h_state[s], h_actions[h], gamma_t, delta_h)
            for s in range(num_seekers):
                delta_s = reward_s + gamma * critic_seek(s_state[s]) - critic_seek(old_s_state[s])
                critic_seek.update(old_s_state[s], delta_s)
                actor_seek.update(old_s_state[s], s_actions[s], gamma_t, delta_s)
            # swap states
            old_h_state = h_state
            old_s_state = s_state
            gamma_t *= gamma
            if done:
                break

        T = len(R_h) - 1
        for t in range(T):
            G_h = 0
            G_s = 0
            for k in range(t+1, T+1):
                G_h += gamma ** (k - t - 1) * R_h[k]
                G_s += gamma ** (k - t - 1) * R_s[k]
            # if t == 0:
            if t == T // 2:
                G0_hide.append(G_h)
                G0_seek.append(G_s)
    return G0_hide, G0_seek




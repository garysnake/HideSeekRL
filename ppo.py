import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim_list, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)
        self.state_dim_list = state_dim_list
        self.state_dim = state_dim

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )

    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        state = self.convert_one_hot(state)
        state = torch.from_numpy(state).float() 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, state, action):
        action_logprobs = []
        for a in range(len(action)):
            action_probs = self.action_layer(state[a])
            dist = Categorical(action_probs)
            action_logprobs.append(dist.log_prob(action[a]))
        torch.tensor(action_logprobs)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def convert_one_hot(self, s):
        one_hot = np.zeros(self.state_dim)
        for a in range(len(s)):
            agent_coord = s[a]
            if -100 in agent_coord:
                continue
            index = agent_coord[0] * self.state_dim_list[2] + agent_coord[1]
            index += a * self.state_dim_list[1] * self.state_dim_list[2]
            one_hot[index] = 1
        return one_hot




class PPO:
    def __init__(self, state_dim_list, action_dim, hidden_layer, lr, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.state_dim_list = state_dim_list
        state_dim = np.prod(state_dim_list)

        self.policy = ActorCritic(state_dim_list, state_dim, action_dim, hidden_layer)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = lr)
        self.policy_old = ActorCritic(state_dim_list, state_dim, action_dim, hidden_layer)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss = nn.MSELoss()

        self.state_dim = state_dim

    def update(self, S, A, R_gamma, old_logprobs):
        
        # Normalizing the rewards:
        R_gamma = torch.tensor(R_gamma)
        if len(S) > 1:
            R_gamma = R_gamma.unsqueeze(0)
        else:
            R_gamma = R_gamma

        S = [torch.FloatTensor(self.convert_one_hot(s)) for s in S]
        A = [torch.FloatTensor([a]) for a in A]


        # Convert list to tensor
        old_states = torch.stack(S).detach()
        # old_actions = torch.stack(A).detach()
        old_actions = A
        old_logprobs = torch.stack(old_logprobs).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # DEBUG
            # print(old_states, old_actions, old_logprobs)
            # print("----")
            # print(logprobs, state_values, dist_entropy)
            # print(old_states.shape, old_actions.shape, old_logprobs.shape)
            # print("----")
            # print(logprobs.shape, state_values.shape, dist_entropy.shape)

            # Finding the ratio (pi_theta / pi_theta__old):
            logprob_list = []
            for l in logprobs:
                logprob_list.append(l.item())
            logprobs = torch.tensor(logprob_list)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss:
            advantages = R_gamma - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
 
            loss = -torch.min(surr1, surr2) + 0.5*self.loss(state_values.unsqueeze(0), R_gamma) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

            
    def convert_one_hot(self, s):
        one_hot = np.zeros(self.state_dim)
        for a in range(len(s)):
            agent_coord = s[a]
            if -100 in agent_coord:
                continue
            index = agent_coord[0] * self.state_dim_list[2] + agent_coord[1]
            index += a * self.state_dim_list[1] * self.state_dim_list[2]
            one_hot[index] = 1
        return one_hot



def train_ppo(grid_world, num_episodes, gamma=.9):
    lr = 0.002
    K_epochs = 4
    eps_clip = 0.2
    hidden_layer = 32

    state_dim_list = grid_world.state_dim_list
    num_actions = grid_world.env_spec.nA

    ppo_hide = PPO(state_dim_list, num_actions, hidden_layer, lr, gamma, K_epochs, eps_clip)
    ppo_seek = PPO(state_dim_list, num_actions, hidden_layer, lr, gamma, K_epochs, eps_clip)
    

    num_hiders = grid_world.numHider
    num_seekers = grid_world.numSeeker
    G0_hide = []
    G0_seek = []

    for episode in range(num_episodes):
        timestep = 0
        gamma_t = 1
        if episode % 10 == 0:
            print("\tEpisode", episode)
        h_state, s_state = grid_world.reset()
        #  Accumulating discounted R
        Rh_disc = 0
        Rs_disc = 0

        R_h = [np.NINF] # Padding for index adjustment
        R_s = [np.NINF] # Padding for index adjustment

        # Generate an episode
        while True:
            h_actions = []
            h_log_probs = []
            for h in range(num_hiders):
                h_a, h_log_prob = ppo_hide.policy_old.act(h_state[h])
                h_actions.append(h_a)
                h_log_probs.append(h_log_prob)
            s_actions = []
            s_log_probs = []
            for s in range(num_seekers):
                s_a, s_log_prob = ppo_seek.policy_old.act(s_state[s])
                s_actions.append(s_a)
                s_log_probs.append(s_log_prob)
                
            ### Produce Behavior Trace ###
            if episode % 50 == 0:
                grid_world.save_world("data/ppo_big_wall" + "_%d-%d" % (num_hiders, num_seekers) + ".txt", episode)
            
            h_state, s_state, reward_h, reward_s, done = grid_world.step(h_actions, s_actions)
            R_h.append(reward_h)
            R_s.append(reward_s)

            #  Calculated the discounted Reward
            Rh_disc = Rh_disc * gamma + reward_h
            Rs_disc = Rs_disc * gamma + reward_s
            Rh_disc_list = [Rh_disc * gamma + reward_h] * num_hiders
            Rs_disc_list = [Rs_disc * gamma + reward_s] * num_seekers

            ppo_hide.update(h_state, h_actions, Rh_disc_list, h_log_probs)
            ppo_seek.update(s_state, s_actions, Rs_disc_list, s_log_probs)
            
            if done:
                break
            timestep += 1

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




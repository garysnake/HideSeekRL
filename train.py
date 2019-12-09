from gridworld import GridWorld_hybrid_state, GridWorld_coord
import numpy as np
import torch
import torch.nn as nn

# --- Code for running random policy as a baseline ---

class RandomPolicy(object):
    def __init__(self, nA, p=None):
        self.p = p if p is not None else np.array([1/nA]*nA)

    def action_prob(self, state, action=None):
        return self.p[action]

    def action(self, state):
        return np.random.choice(len(self.p), p=self.p)

def random_baseline(grid_world, num_episodes):
    """
    Run random policy and return discounted monte carlo rewards at t=0
    """
    num_hiders = grid_world.numHider
    num_seekers = grid_world.numSeeker
    behavior_policy = RandomPolicy(5)
    T = grid_world.eps_len - 1
    G0s_hide = []
    G0s_seek = []
    for e in range(num_episodes):
        init_h, init_s = grid_world.reset()
        hider_states = [init_h]
        seeker_states = [init_s]
        hider_rewards = [np.NINF]
        seeker_rewards = [np.NINF]
        hider_actions = []
        seeker_actions = []
        while True:
            h_action = behavior_policy.action(None)
            s_action = behavior_policy.action(None)
            # print("Hider trying", h_action, "Seeker trying", s_action)
            hs, ss, hr, sr, done = grid_world.step([h_action] * num_hiders, [s_action] * num_seekers)
            hider_states.append(hs)
            seeker_states.append(ss)
            hider_actions.append(h_action)
            seeker_actions.append(s_action)
            hider_rewards.append(hr)
            seeker_rewards.append(sr)
            # grid_world.print_world() # print world representation
            # print("^^^ Reward got from above", hr, sr)
            if done:
                break
        # Calculate and append the monte carlo return at t_0
        G_hide = 0
        G_seek = 0
        # for k in range(1, T + 1):
        for k in range(T // 2, T + 1):
            G_hide += grid_world.env_spec.gamma ** (k) * hider_rewards[k]
            G_seek += grid_world.env_spec.gamma ** (k) * seeker_rewards[k]
        G0s_hide.append(G_hide)
        G0s_seek.append(G_seek)
    return G0s_hide, G0s_seek

# --- Code for running SARSA --- (TODO: not correct yet)

class StateActionFeatureVector():
    def __init__(self, dim_list, num_actions):
        self.state_dims = list(dim_list)
        self.num_actions = num_actions
        self.state_dims.append(num_actions)
        self.feature_vector_length = np.prod(self.state_dims)

    def __call__(self, s, done, a) -> np.array:
        one_hot = np.zeros(self.feature_vector_length)
        if done:
            return one_hot
        for agent in range(len(s)):
            agent_coord = s[agent]
            if -100 in agent_coord:
                continue
            index = agent_coord[0] * self.state_dims[2] + agent_coord[1]
            index += agent * self.state_dims[1] * self.state_dims[2]
            index *= self.num_actions
            index += a
            one_hot[index] = 1

        return one_hot

def SarsaLambda(grid_world, num_episode, gamma=.9, lam=.8, alpha=.01):

    def epsilon_greedy_policy(s, done, w, X, num_agents, epsilon=.0):
        """
        modified to return list of actions
        """
        nA = grid_world.env_spec.nA
        list_actions = []
        for n in range(num_agents):
            Q = [np.dot(w, X(s[n],done,a)) for a in range(nA)]
            if np.random.rand() < epsilon:
                list_actions.append(np.random.randint(nA))
            else:
                list_actions.append(np.argmax(Q))

    # Create x's
    dim_list = grid_world.state_dim_list
    X_h = StateActionFeatureVector(dim_list, grid_world.env_spec.nA)
    X_s = StateActionFeatureVector(dim_list, grid_world.env_spec.nA)

    dimension = X_h.feature_vector_length
    w_h = np.zeros(dimension)
    w_s = np.zeros(dimension)

    num_hiders = grid_world.numHider
    num_seekers = grid_world.numSeeker
    G0_hide = []
    G0_seek = []
    for episode in range(num_episode):
        if episode % 10 == 0:
            print("\tEpisode", episode)
        S_h, S_s = grid_world.reset()
        done = False
        A_h = epsilon_greedy_policy(S_h, done, w_h, X_h, num_hiders)
        A_s = epsilon_greedy_policy(S_s, done, w_s, X_s, num_seekers)
        x_h = X_h(S_h, done, A_h)
        x_s = X_s(S_s, done, A_h)
        z_h = np.zeros(dimension)
        z_s = np.zeros(dimension)
        Q_old_h = 0.
        Q_old_s = 0.

        # G0 stuff
        R_h = [np.NINF]
        R_s = [np.NINF]

        while True:
            S_new_h, S_new_s, reward_h, reward_s, done = grid_world.step()
            R_h.append(reward_h)
            R_s.append(reward_s)
            A_new_h = epsilon_greedy_policy(S_new_h, done, w_h, X_h, num_hiders)
            A_new_s = epsilon_greedy_policy(S_new_s, done, w_s, X_s, num_seekers)
            # TODO: After this line it's not working
            x_new_h = X_h(S_new_h, done, A_new_h)
            x_new_s = X_s(S_new_s, done, A_new_s)
            Q_h = np.dot(w_h, x_h)
            Q_s = np.dot(w_s, x_s)
            Q_new_h = np.dot(w_h, x_new_h)
            Q_new_s = np.dot(w_s, x_new_s)
            delta_h = reward_h + gamma * Q_new_h - Q_h
            delta_s = reward_s + gamma * Q_new_s - Q_s
            z_h = gamma * lam * z_h + (1 - alpha * gamma * lam * np.dot(z_h, x_h))*x_h
            z_s = gamma * lam * z_s + (1 - alpha * gamma * lam * np.dot(z_s, x_s))*x_s
            w_h += alpha *  (delta_h + Q_h - Q_old_h) * z_h - alpha *(Q_h - Q_old_h) * x_h
            w_s += alpha *  (delta_s + Q_s - Q_old_s) * z_s - alpha *(Q_s - Q_old_s) * x_s
            Q_old_h = Q_new_h
            Q_old_s = Q_new_s
            x_h = x_new_h
            x_s = x_new_s
            A_h = A_new_h
            A_s = A_new_s
            if done:
                break
        T = len(S_h) - 1
        for t in range(T):
            G_h = 0
            G_s = 0
            for k in range(t + 1, T + 1):
                G_h += gamma ** (k - t - 1) * R_h[k]
                G_s += gamma ** (k - t - 1) * R_s[k]
            if t == T // 2:
                G0_hide.append(G_h)
                G0_seek.append(G_s)
    return G0_hide, G0_seek

# --- Code for running REINFORCE ---

class pi_nn(object):
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.state_size = np.prod(state_dims)
        self.network = nn.Sequential(
            nn.Linear(self.state_size, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
            nn.Softmax(dim=-1))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)

    def convert_one_hot(self, s):
        one_hot = np.zeros(self.state_size)
        for a in range(len(s)):
            agent_coord = s[a]
            if -100 in agent_coord:
                continue
            index = agent_coord[0] * self.state_dims[2] + agent_coord[1]
            index += a * self.state_dims[1] * self.state_dims[2]
            one_hot[index] = 1
        return one_hot

    def __call__(self,s) -> int:
        s = self.convert_one_hot(s)
        prods = self.network(torch.FloatTensor(s)).detach().numpy()
        action = np.random.choice(range(self.num_actions), p=prods)
        return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.optimizer.zero_grad()
        s = self.convert_one_hot(s)
        output = self.network(torch.FloatTensor(s)).view(1, -1)
        target = torch.tensor([a])
        loss = gamma_t * delta * self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return None

class Baseline(object):
    """ constant baseline """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class v_nn(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.state_dims = state_dims
        self.state_size = np.prod(state_dims)
        self.network = nn.Sequential(
            nn.Linear(self.state_size, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)

    def convert_one_hot(self, s):
        one_hot = np.zeros(self.state_size)
        for a in range(len(s)):
            agent_coord = s[a]
            if -100 in agent_coord:
                continue
            index = agent_coord[0] * self.state_dims[2] + agent_coord[1]
            index += a * self.state_dims[1] * self.state_dims[2]
            one_hot[index] = 1
        return one_hot

    def __call__(self,s) -> float:
        s = self.convert_one_hot(s)
        return self.network(torch.FloatTensor(s))

    def update(self,s,G):
        self.optimizer.zero_grad()
        s = self.convert_one_hot(s)
        output = self.network(torch.FloatTensor(s))
        loss = self.criterion(output, torch.FloatTensor([G]))
        loss.backward()
        self.optimizer.step()
        return None

def REINFORCE(grid_world, num_episodes, gamma=.9):
    """
    outputs lists of G_0 for every episode
    """
    # TODO: modify for gridworld
    
    state_dim = grid_world.state_dim_list
    num_actions = grid_world.env_spec.nA
    alpha = 3e-4
    pi_hide = pi_nn(state_dim, num_actions, alpha)
    pi_seek = pi_nn(state_dim, num_actions, alpha)
    v_hide = v_nn(state_dim, alpha)
    v_seek = v_nn(state_dim, alpha)

    num_hiders = grid_world.numHider
    num_seekers = grid_world.numSeeker
    G0_hide = []
    G0_seek = []
    for episode in range(num_episodes):
        if episode % 10 == 0:
            print("\tEpisode", episode)
        h_state, s_state = grid_world.reset()
        S_h = [h_state]
        R_h = [np.NINF] # Padding for index adjustment
        A_h = []

        S_s = [s_state]
        R_s = [np.NINF] # Padding for index adjustment
        A_s = []

        # Generate an episode
        while True:
            h_actions = []
            for h in range(num_hiders):
                h_actions.append(pi_hide(h_state[h]))
            s_actions = []
            for s in range(num_seekers):
                s_actions.append(pi_seek(s_state[s]))
            h_state, s_state, h_r, s_r, done = grid_world.step(h_actions, s_actions)
            S_h.append(h_state)
            R_h.append(h_r)
            A_h.append(h_actions)
            S_s.append(s_state)
            R_s.append(s_r)
            A_s.append(s_actions)
            if done:
                break
        T = len(S_h) - 1
        for t in range(T):
            G_h = 0
            G_s = 0
            for k in range(t+1, T+1):
                G_h += gamma ** (k - t - 1) * R_h[k]
                G_s += gamma ** (k - t - 1) * R_s[k]
            for h in range(num_hiders):
                delta_h = G_h - v_hide(S_h[t][h])
                v_hide.update(S_h[t][h], G_h)
                pi_hide.update(S_h[t][h], A_h[t][h], gamma ** t, delta_h)
            for s in range(num_seekers):
                delta_s = G_s - v_seek(S_s[t][s])
                v_seek.update(S_s[t][s], G_s)
                pi_seek.update(S_s[t][s], A_s[t][s], gamma ** t, delta_s)
            # if t == 0:
            if t == T // 2:
                G0_hide.append(G_h)
                G0_seek.append(G_s)
    return G0_hide, G0_seek

# -- Code for running Actor-Critic

class actor_nn(nn.Module):
    def __init__(self, state_dims, num_actions, alpha):
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.state_size = np.prod(state_dims)
        self.network = nn.Sequential(
            nn.Linear(self.state_size, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
            nn.Softmax(dim=-1))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)

    def convert_one_hot(self, s):
        one_hot = np.zeros(self.state_size)
        for a in range(len(s)):
            agent_coord = s[a]
            if -100 in agent_coord:
                continue
            index = agent_coord[0] * self.state_dims[2] + agent_coord[1]
            index += a * self.state_dims[1] * self.state_dims[2]
            one_hot[index] = 1
        return one_hot

    def __call__(self,s) -> int:
        s = self.convert_one_hot(s)
        prods = self.network(torch.FloatTensor(s)).detach().numpy()
        action = np.random.choice(range(self.num_actions), p=prods)
        return action

    def update(self, s, a, gamma_t, delta):
        self.optimizer.zero_grad()
        s = self.convert_one_hot(s)
        output = self.network(torch.FloatTensor(s)).view(1, -1)
        target = torch.tensor([a])
        loss = gamma_t * delta * self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return None

class critic_nn(Baseline):
    def __init__(self, state_dims, alpha):
        self.state_dims = state_dims
        self.state_size = np.prod(state_dims)
        self.network = nn.Sequential(
            nn.Linear(self.state_size, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)

    def convert_one_hot(self, s):
        one_hot = np.zeros(self.state_size)
        for a in range(len(s)):
            agent_coord = s[a]
            if -100 in agent_coord:
                continue
            index = agent_coord[0] * self.state_dims[2] + agent_coord[1]
            index += a * self.state_dims[1] * self.state_dims[2]
            one_hot[index] = 1
        return one_hot

    def __call__(self,s) -> float:
        s = self.convert_one_hot(s)
        return self.network(torch.FloatTensor(s))

    def update(self, s, delta):
        self.optimizer.zero_grad()
        s = self.convert_one_hot(s)
        output = self.network(torch.FloatTensor(s))
        loss = self.criterion(output, torch.FloatTensor([delta]))
        loss.backward()
        self.optimizer.step()
        return None

def actor_critic(grid_world, num_episodes, gamma=.9):
    alpha = 3e-4
    state_dim = grid_world.state_dim_list
    num_actions = grid_world.env_spec.nA
    actor_hide = pi_nn(state_dim, num_actions, alpha)
    actor_seek = pi_nn(state_dim, num_actions, alpha)
    critic_hide = v_nn(state_dim, alpha)
    critic_seek = v_nn(state_dim, alpha)
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
                grid_world.save_world("data/ac_two_walls.txt", episode)
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
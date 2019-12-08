from gridworld import GridWorld_hybrid_state, GridWorld_one_hot
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

# --- Code for running SARSA --- (TODO)

class StateActionFeatureVector():
    def __init__(self, dim_list, num_actions):
        self.dim_list = dim_list
        self.dim_list.append(num_actions)
        self.feature_vector_length = np.prod(dim_list) * num_actions

    def find_flat_vector_coord(self, s, a):
        vector_coord = list(s) + [a]
        index_padding = self.feature_vector_length
        index = 0
        for i in range(len(vector_coord)):
            index_padding /= self.dim_list[i]
            index += vector_coord[i] * index_padding
        return int(index)

    def __call__(self, s, done, a) -> np.array:
        one_hot = np.zeros(self.feature_vector_length)
        if done:
            return one_hot
        one_hot[self.find_flat_vector_coord(s, a)] = 1
        return one_hot

def SarsaLambda(grid_world, gamma, lam, alpha, num_episode):

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = grid_world.env_spec.nA
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    # Create x's
    dim_list = grid_world.state_dim_list
    X_hide = StateActionFeatureVector(dim_list, grid_world.env_spec.nA)
    X_seek = StateActionFeatureVector(dim_list, grid_world.env_spec.nA)

    dimension = X.feature_vector_len()
    w = np.zeros(dimension)

    num_hiders = grid_world.numHider
    num_seekers = grid_world.numSeeker
    G0s_hide = []
    G0s_seek = []
    for episode in range(num_episode):
        S = grid_world.reset()
        done = False
        A = epsilon_greedy_policy(S, done, w)
        x = X(S, done, A)
        z = np.zeros(dimension)
        Q_old = 0.
        
        # print("episode:", episode)
        # R = 0

        while True:
            # env.render()
            S_new, reward, done, info = env.step(A)
            A_new = epsilon_greedy_policy(S_new, done, w)
            x_new = X(S_new, done, A_new)
            Q = np.dot(w, x)
            Q_new = np.dot(w, x_new)
            delta = reward + gamma*Q_new - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x))*x
            w += alpha *  (delta + Q - Q_old) * z - alpha *(Q - Q_old) * x
            Q_old = Q_new
            x = x_new
            A = A_new
            if done:
                break
    return w

# --- Code for running REINFORCE ---

class pi_nn(object):
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.num_actions = num_actions
        self.network = nn.Sequential(
            nn.Linear(state_dims, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
            nn.Softmax(dim=-1))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)

    def __call__(self,s) -> int:
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
        self.network = nn.Sequential(
            nn.Linear(state_dims, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=alpha)

    def __call__(self,s) -> float:
        return self.network(torch.FloatTensor(s))

    def update(self,s,G):
        self.optimizer.zero_grad()
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
    
    state_dim = grid_world.state_dim
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
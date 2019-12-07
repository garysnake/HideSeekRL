import numpy as np
from env import EnvSpec, Env
from enum import Enum

class GridWorld(Env):
    def __init__(self, numX, numY, numHider, numSeeker, hideVis, seekVis, eps_len):
        self.env_spec = EnvSpec(numX * numY, 5, 0.9)
        self.eps_len = eps_len
        self.step_count = 0
        self.numHider = numHider
        self.numSeeker = numSeeker
        self.numX = numX
        self.numY = numY
        # Note: vis are "radius" of cube of vision
        assert hideVis*2+1 <= numX and hideVis*2+1 <= numY, "hideVis too big"
        assert seekVis*2+1 <= numX and seekVis*2+1 <= numY, "seekVis too big"
        self.hideVis = hideVis
        self.seekVis = seekVis
        self.world = np.zeros((self.numX, self.numY))
        self.init_world()

    def print_world(self):
        print("\nBegin GridWorld")
        print("-----------------------------------")
        for i in range(self.numX):
            item = ""
            for j in range(self.numY):
                obj = self.world[i][j]
                if obj == 0:
                    item += ". "
                elif obj == 1:
                    item += "W "
                elif obj == 2:
                    item += "H "
                else:
                    item += "S "
            print(item)
        print("-----------------------------------")
        print("End GridWorld\n")

    def reset(self):
        self.step_count = 0
        self.init_world()
        h_list, _ = self.get_agent_state_reward(self.list_hiders, self.hideVis, agent_type="hide")
        s_list, _ = self.get_agent_state_reward(self.list_seekers, self.seekVis, agent_type="seek")
        return h_list, s_list

    def init_world(self):
        """
        0 = normal state
        1 = wall
        2 = hide agent
        3 = seek agent
        4 = unknown (not used until masked)
        5 = self position (not used until masked)
        """
        self.numO = 6
        # wall creation. Currently "game of tag" formation (no random walls)
        for i in range(self.numX):
            for j in range(self.numY):
                if i == 0 or i == self.numX - 1 or j == 0 or j == self.numY - 1:
                    self.world[i][j] = 1
                else:
                    self.world[i][j] = 0
        # Extra walls
        wall_list = [[2,2], [4,4], [6,6]]
        for wall in wall_list:
            self.world[tuple(wall)] = 1
        def get_rand_coord():
            # Utility to find random agent placement
            x_coord = np.random.randint(1, self.numX - 1)
            y_coord = np.random.randint(1, self.numY - 1)
            return x_coord, y_coord
        # random hider placement
        self.list_hiders = []
        for i in range(self.numHider):
            x_coord, y_coord = get_rand_coord()
            while self.world[x_coord][y_coord] != 0:
                x_coord, y_coord = get_rand_coord()
            self.world[x_coord][y_coord] = 2
            self.list_hiders.append([x_coord, y_coord])
        # random seeker placement
        self.list_seekers = []
        for i in range(self.numSeeker):
            x_coord, y_coord = get_rand_coord()
            while self.world[x_coord][y_coord] != 0:
                x_coord, y_coord = get_rand_coord()
            self.world[x_coord][y_coord] = 3
            self.list_seekers.append([x_coord, y_coord])
    
    def get_agent_state_reward(self, list_agents, agentVis, agent_type="hide"):
        """
        return list of states for each inputted agent as well as reward
        """
        if agent_type =="hide":
            enemy_type = 3
            list_agents = self.list_hiders
        else:
            enemy_type = 2
            list_agents = self.list_seekers
        enemies_spotted = 0
        agent_states = []
        # Get visibility for each agent
        for a in list_agents:
            agent_world = np.copy(self.world)
            for x in range(self.numX):
                for y in range(self.numY):
                    if (x < a[0] - agentVis or x > a[0] + agentVis) \
                        or (y < a[1] - agentVis or y > a[1] + agentVis):
                        agent_world[x][y] = 4
            self.patchShadow(a[0], a[1], agentVis, agent_world)
            agent_states.append(agent_world)
        # Stack visibilities into common agent world
        agent_common_view = np.ones((self.world.shape))
        agent_common_view.fill(4)
        for i in range(self.numX):
            for j in range(self.numY):
                for vis_state in agent_states:
                    if vis_state[i][j] != 4:
                        agent_common_view[i][j] = vis_state[i][j]
                        break
        print("Common view for", agent_type, "\n", agent_common_view)
        # get list of states with agent position in one-hot form
        agent_states = []
        for a in list_agents:
            agent_state = np.zeros(self.numX * self.numY * self.numO)
            for i in range(self.numX):
                for j in range(self.numY):
                    if agent_world[i][j] == enemy_type:
                        enemies_spotted += 1
                    if i == a[0] and j == a[1]:
                        idx = i * self.numY * self.numO + j * self.numO + 5
                    else:
                        idx = i * self.numY * self.numO + j * self.numO + agent_common_view[i][j]
                    agent_state[int(idx)] = 1
            agent_states.append(agent_state)
        if agent_type == "hide":
            reward = 1 if enemies_spotted == 0 else -1
        else:
            reward = 1 if enemies_spotted == 1 else -1
        return agent_states, reward

    def patchShadow(self, x_coord, y_coord, vis, world):
        """
        Approximation of field of view and quadrant-based wall shadows
        """
        def isShadow(x, y):
            return (world[x][y] == 4 or world[x][y] == 1)

        rightBound = min(x_coord+vis+1, world.shape[0])
        leftBound = max(x_coord-vis-1, -1)
        upperBound = min(y_coord+vis+1, world.shape[1])
        lowerBound = max(y_coord-vis-1, -1)

        # invert cross indices to not mess up quadrant calculations
        for x in range(leftBound, rightBound):
            if world[x][y_coord] == 1:
                world[x][y_coord] = 420
        for y in range(lowerBound, upperBound):
            if world[x_coord][y] == 1:
                world[x_coord][y] = 420

        for x in range(x_coord+1, rightBound): # Upper right
            for y in range(y_coord+1, upperBound):
                if isShadow(x-1,y) or isShadow(x,y-1) or isShadow(x-1,y-1):
                    world[x][y] = 4
        for x in range(x_coord-1, leftBound, -1): # Lower left
            for y in range(y_coord-1, lowerBound,-1):
                if isShadow(x+1,y) or isShadow(x,y+1) or isShadow(x+1,y+1):
                    world[x][y] = 4
        for x in range(x_coord+1, rightBound): # Lower right
            for y in range(y_coord-1, lowerBound,-1):
                if isShadow(x-1,y) or isShadow(x,y+1) or isShadow(x-1,y+1):
                    world[x][y] = 4
        for x in range(x_coord-1, leftBound, -1): # Upper left
            for y in range(y_coord+1, upperBound):
                if isShadow(x+1,y) or isShadow(x,y-1) or isShadow(x+1,y-1):
                    world[x][y] = 4
        # re-invert cross indices to do cross vision
        for x in range(leftBound, rightBound):
            if world[x][y_coord] == 420:
                world[x][y_coord] = 1
        for y in range(lowerBound, upperBound):
            if world[x_coord][y] == 420:
                world[x_coord][y] = 1
        for x in range(x_coord - 1, leftBound, -1): # Horizontal cross left
            if isShadow(x + 1, y_coord):
                world[x][y_coord] = 4
        for x in range(x_coord + 1, rightBound): # Horizontal cross right
            if isShadow(x - 1, y_coord):
                world[x][y_coord] = 4
        for y in range(y_coord - 1, lowerBound, -1): # Vertical cross down
            if isShadow(x_coord, y + 1):
                world[x_coord][y] = 4
        for y in range(y_coord + 1, upperBound): # Vertical cross up
            if isShadow(x_coord, y - 1):
                world[x_coord][y] = 4

    def can_move(self, destination):
        return self.world[destination[0]][destination[1]] == 0

    def take_actions(self, agent_actions, agent_type="hide"):
        """
        Take a list of actions depending on agent type
        """
        if agent_type == "hide":
            list_agents = self.list_hiders
            agent_type = 2
        else:
            list_agents = self.list_seekers
            agent_type = 3
        for agent_idx in range(len(list_agents)):
            action = agent_actions[agent_idx]
            start_coord = list_agents[agent_idx]
            dest_coord = list(start_coord)
            if action == 0:
                pass
            elif action == 1:
                dest_coord[0] -= 1
            elif action == 2:
                dest_coord[1] += 1
            elif action == 3:
                dest_coord[0] += 1
            elif action == 4:
                dest_coord[1] -= 1

            if self.can_move(dest_coord):
                self.world[tuple(start_coord)] = 0
                self.world[tuple(dest_coord)] = agent_type
                list_agents[agent_idx] = list(dest_coord)

    def step(self, hider_actions, seeker_actions):
        """
        Input: lists of actions to take for hiders and seekers
        Output: hider_states, seeker_states, hider_reward, seeker_reward, done
        """
        assert self.step_count < self.eps_len, "Cannot step beyond end of episode"
        self.step_count += 1
        self.take_actions(hider_actions, agent_type="hide")
        self.take_actions(seeker_actions, agent_type="seek")
        h_list, h_r = self.get_agent_state_reward(self.list_hiders, self.hideVis, agent_type="hide")
        s_list, s_r = self.get_agent_state_reward(self.list_seekers, self.seekVis, agent_type="seek")
        return h_list, s_list, h_r, s_r, self.step_count == self.eps_len
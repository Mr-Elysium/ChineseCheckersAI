
from copy import copy
from chinesecheckers_env.env.chinesecheckers_utils import check_winner

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Dict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

NUM_PINS = 10
BOARD_SIZE = np.floor(np.sqrt(NUM_PINS * 2))
BOARD_HIST = 3
MAX_MOVES = 1000
NUM_PLAYERS = 2

class ChineseCheckersEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "chinese_checkers_environment",
    }

    def __init__(self, render_mode: str = "human"):
        self.agents = [f"players_{i}" for i in range(NUM_PLAYERS)]
        self.possible_agents = self.agents[:]
        self.move = 0
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.action_space_dim = (2 * BOARD_SIZE + 1) ** 4
        self.observation_space_dim = 2 * (2 * BOARD_SIZE + 1)
        self.action_spaces = {agent: Discrete(self.action_space_dim) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Dict({
                "observation": MultiDiscrete([self.observation_space_dim] * 3),
                "action_mask": MultiDiscrete([self.observation_space_dim] * 3)
            })
            for agent in self.possible_agents
        }

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.move = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0

        self.state[agent] = action

    def render(self):
        print(f"{self.board} \n")

    def observation_space(self, agent):
        return MultiDiscrete([BOARD_SIZE, BOARD_SIZE, BOARD_HIST] * 3)

    def action_space(self, agent):
        return MultiDiscrete([BOARD_SIZE, BOARD_SIZE, 4])
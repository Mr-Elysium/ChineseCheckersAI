
from copy import copy

from pettingzoo.utils.env import AgentID, ObsType

from chinesecheckers_utils import action_to_move, get_legal_move_mask
from chinesecheckers_game import ChineseCheckers, Direction, Move, Position

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Dict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import matplotlib.pyplot as plt

NUM_PINS = 10
BOARD_SIZE = int(np.floor(np.sqrt(NUM_PINS * 2)))
BOARD_HIST = 3
MAX_MOVES = 1000
NUM_PLAYERS = 2

class ChineseCheckersEnv(AECEnv):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "name": "chinese_checkers_environment",
    }

    def __init__(self, render_mode: str = "rgb_array"):
        self.max_moves = MAX_MOVES

        self.agents = [f"players_{i}" for i in range(NUM_PLAYERS)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.move = 0

        self.action_space_dim = (2 * BOARD_SIZE + 1) ** 4
        self.observation_space_dim = ((2 * BOARD_SIZE + 1) ** 2) * 2
        self.action_spaces = {agent: Discrete(self.action_space_dim) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: Dict({
                "observation": Discrete(self.observation_space_dim),
                "action_mask": Discrete(self.action_space_dim)
            })
            for agent in self.possible_agents
        }

        self.agent_selection = None

        self.window_size = 1000
        self.render_mode = render_mode

        self.game = ChineseCheckers(10, render_mode=render_mode)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.game.init_game()
        self.move = 0
        self.winner = None

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player = self.agent_name_mapping[agent]

        action = int(action)

        move = action_to_move(action, BOARD_SIZE)
        move_result = self.game.move(player, move)

        if self.game.did_player_win(player):
            self.terminations = {
                agent: self.game.is_game_over()
            }
            for a in self.agents:
                self.rewards[a] = 10 if a == agent else -1
            self.winner = agent
        elif move is None:
            self.rewards[agent] = -1000
        else:
            if move.move_to in self.game.get_home_coordinates(player):
                self.rewards[agent] += 0.1

        self._accumulate_rewards()
        self._clear_rewards()

        if self._agent_selector.is_last():
            self.truncations = {
                agent: self.move >= self.max_moves for agent in self.agents
            }

        self.move += 1
        self.agent_selection = self._agent_selector.next()

    def render(self):
        return self.game.render()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def observe(self, agent):
        player = self.agent_name_mapping[agent]

        def pins_to_board(pin_coords):
            board = np.zeros((BOARD_SIZE, BOARD_SIZE))
            for pos in pin_coords:
                board[pos.x][pos.y] = 1
            return board

        player_pins = self.game.get_player_coordinates(player)
        if player == 0:
            opponent_pins = self.game.get_player_coordinates(1)
        else:
            opponent_pins = self.game.get_player_coordinates(0)

        observation = np.stack([pins_to_board(player_pins), pins_to_board(opponent_pins)])

        observation = observation.flatten()

        return {
            "observation": observation,
            "action_mask": get_legal_move_mask(self.game, player, BOARD_SIZE)
        }


    def action_space(self, agent):
        return self.action_spaces[agent]

if __name__ == "__main__":
    env = ChineseCheckersEnv()
    env.reset()

    frame = env.render()
    plt.imshow(frame)
    plt.tight_layout()
    plt.savefig("test.png")
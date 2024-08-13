from chinesecheckers.chinesecheckers_utils import action_to_move, get_legal_move_mask
from chinesecheckers.chinesecheckers_game import ChineseCheckers

import numpy as np
from gymnasium.spaces import Discrete, Dict, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

import matplotlib.pyplot as plt

NUM_PINS = 10
TRIANGLE_SIZE = int(np.floor(np.sqrt(NUM_PINS * 2)))
BOARD_SIZE = 2 * TRIANGLE_SIZE + 1
BOARD_HIST = 3
MAX_MOVES = 1000
NUM_PLAYERS = 2

WIN_REWARD = 100
LOSE_REWARD = -100
HOME_JUMP_REWARD = 5
FORWARD_JUMP_REWARD = 0.05
INVALID_MOVE_REWARD = -1000

class ChineseCheckersEnv(AECEnv):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "name": "chinese_checkers_2v2",
    }

    def __init__(self, render_mode: str = "rgb_array"):
        super().__init__()
        self.max_moves = MAX_MOVES

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.move_iter = 0

        self.action_space_size = BOARD_SIZE ** 4
        self.action_spaces = {agent: Discrete(self.action_space_size) for agent in self.agents}

        self.observation_space_shape = (BOARD_SIZE, BOARD_SIZE, 2)

        self.observation_spaces = {
            agent: Dict({
                "observation": Box(shape=self.observation_space_shape, dtype=np.int8, low=0, high=1),
                "action_mask": Box(shape=(self.action_space_size,), dtype=np.int8, low=0, high=1)
            })
            for agent in self.possible_agents
        }

        self.window_size = 1000
        self.render_mode = render_mode

        self.game = ChineseCheckers(NUM_PINS, render_mode=render_mode)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.game.init_game()
        self.move_iter = 0
        self.winner = None

        self._agent_selector = agent_selector(self.agents)

        self.agent_selection = self._agent_selector.reset()

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player = self.agent_name_mapping[agent]

        action = int(action)

        move = action_to_move(action, BOARD_SIZE)
        move_result = self.game.move(player, move)

        next_agent = self._agent_selector.next()

        if self.game.did_player_win(player):
            self.terminations = {a: self.game.is_game_over() for a in self.agents}
            self.rewards[agent] += WIN_REWARD
            self.rewards[next_agent] += LOSE_REWARD
            self.winner = agent
        elif move_result is None:
            self.rewards[agent] = INVALID_MOVE_REWARD
        else:
            if move.move_to in self.game.get_home_coordinates(player):
                self.rewards[agent] += HOME_JUMP_REWARD
            if player == 0:
                self.rewards[agent] += max(((move.move_to.x + move.move_to.y) - (move.move_from.x + move.move_from.y)) * FORWARD_JUMP_REWARD, 0)
            elif player == 1:
                self.rewards[agent] += max(((move.move_from.x + move.move_from.y) - (move.move_to.x + move.move_to.y)) * FORWARD_JUMP_REWARD, 0)

        self._accumulate_rewards()

        if self._agent_selector.is_last():
            self.truncations = {
                agent: self.move_iter >= self.max_moves for agent in self.agents
            }

        self.move_iter += 1
        self.agent_selection = next_agent

    def render(self):
        return self.game.render()

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

        action_mask = get_legal_move_mask(self.game, player, BOARD_SIZE)

        return {
            "observation": observation,
            "action_mask": action_mask
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

if __name__ == "__main__":
    env = ChineseCheckersEnv()
    env.reset()

    frame = env.render()
    plt.imshow(frame)
    plt.tight_layout()
    plt.savefig("test.png")
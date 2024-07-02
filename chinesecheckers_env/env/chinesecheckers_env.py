
from copy import copy
from chinesecheckers_env.env.chinesecheckers_utils import check_winner

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv

BOARD_SIZE = 9
BOARD_HIST = 3

class ChineseCheckersEnv(ParallelEnv):
    metadata = {
        "name": "chinese_checkers_environment_v0",
    }

    def __init__(self):

        self.board = None
        self.possible_players = ["player1", "player2"]
        self.starting_board = np.zeros((BOARD_SIZE, BOARD_SIZE, BOARD_HIST), dtype='uint8')
        self.starting_board[:, :, 0] = np.array([[0, 0, 0, 0, 0, 2, 2, 2, 2],
                                       [0, 0, 0, 0, 0, 0, 2, 2, 2],
                                       [0, 0, 0, 0, 0, 0, 0, 2, 2],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 2],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 0, 0, 0, 0, 0]])
        self.timestep = None

    def reset(self, seed=None, options=None):

        self.players = copy(self.possible_players)
        self.timestep = 0
        self.board = self.starting_board

        observations = {
            player: self.board for player in self.players
        }

        infos = {player: {} for player in self.players}

        return observations, infos

    def step(self, actions):
        """"
           0
        1     2
           3
        """
        self.board[:, :, 2] = self.board[:, :, 1]
        self.board[:, :, 1] = self.board[:, :, 0]

        player1_action = actions["player1"]
        player2_action = actions["palyer2"]

        player1_pawn_x = player1_action // 100
        player1_pawn_y = player1_action // 10 % 10
        player1_pawn_direction = player1_action % 10

        self.board[player1_pawn_x, player1_pawn_y, 0] = 0
        if player1_pawn_direction == 0: self.board[player1_pawn_x, player1_pawn_y + 1, 0] = 1
        if player1_pawn_direction == 1: self.board[player1_pawn_x - 1, player1_pawn_y, 0] = 1
        if player1_pawn_direction == 2: self.board[player1_pawn_x + 1, player1_pawn_y, 0] = 1
        if player1_pawn_direction == 3: self.board[player1_pawn_x, player1_pawn_y - 1, 0] = 1

        player2_pawn_x = player2_action // 100
        player2_pawn_y = player2_action // 10 % 10
        player2_pawn_direction = player2_action % 10

        self.board[player2_pawn_x, player2_pawn_y, 0] = 0
        if player2_pawn_direction == 0: self.board[player2_pawn_x, player2_pawn_y + 1, 0] = 1
        if player2_pawn_direction == 1: self.board[player2_pawn_x - 1, player2_pawn_y, 0] = 1
        if player2_pawn_direction == 2: self.board[player2_pawn_x + 1, player2_pawn_y, 0] = 1
        if player2_pawn_direction == 3: self.board[player2_pawn_x, player2_pawn_y - 1, 0] = 1

        terminations = {player: False for player in self.players}
        rewards = {player: 0 for player in self.players}

        if check_winner(self.board[:, :, 0] == 1):
            rewards = {"player1": 1, "palyer2": -1}
            terminations = {player: True for player in self.players}
        elif check_winner(self.board[:, :, 0] == 2):
            rewards = {"player1": -1, "palyer2": 1}
            terminations = {player: True for player in self.players}

        truncations = {player: False for player in self.players}
        if self.timestep > 200:
            rewards = {player: 0 for player in self.players}
            truncations = {player: True for player in self.players}
        self.timestep += 1

        observations = {
            player: self.board for player in self.players
        }

        infos = {player: {} for player in self.players}

        if any(terminations.values()) or all(truncations.values()):
            self.players = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        print(f"{self.board} \n")

    def observation_space(self, agent):
        return MultiDiscrete([BOARD_SIZE, BOARD_SIZE, BOARD_HIST] * 3)

    def action_space(self, agent):
        return Discrete(999)
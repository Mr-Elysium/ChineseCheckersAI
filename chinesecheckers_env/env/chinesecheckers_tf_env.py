import copy
from typing import Any

from tf_agents.typing import types

from chinesecheckers_utils import action_to_move, get_legal_move_mask
from chinesecheckers_game import ChineseCheckers, Direction, Move, Position

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Dict
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import matplotlib.pyplot as plt

import tensorflow as tf
import abc

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories import TimeStep, StepType

NUM_PINS = 10
BOARD_SIZE = int(np.floor(np.sqrt(NUM_PINS * 2)))
BOARD_HIST = 3
MAX_MOVES = 1000
NUM_PLAYERS = 2

class ChineseCheckersEnv(py_environment.PyEnvironment):

    REWARD_WIN = np.asarray(10., dtype=np.float32)
    REWARD_LOSS = np.asarray(-10., dtype=np.float32)
    REWARD_ILLEGAL_MOVE = np.asarray(-1000., dtype=np.float32)

    REWARD_WIN.setflags(write=False)
    REWARD_LOSS.setflags(write=False)
    REWARD_ILLEGAL_MOVE.setflags(write=False)

    def __init__(self, discount: int = 1.0, render_mode: str = 'rgb_array'):
        super(ChineseCheckersEnv, self).__init__()

        self.max_moves = MAX_MOVES

        self.move = 0

        self._discount = np.asarray(discount, dtype=np.float32)

        self.game = ChineseCheckers(10, render_mode=render_mode)

        self._episode_ended = False
        self._states = None

    def action_spec(self):
        return BoundedArraySpec((BOARD_SIZE ** 4,), np.int32, minimum=0, maximum=BOARD_SIZE ** 4, name='action')

    def observation_spec(self):
        return BoundedArraySpec((BOARD_SIZE, BOARD_SIZE), np.int32, minimum=-1, maximum=1)

    def _reset(self, seed=None, options=None):

        self.game.init_game()
        self.move = 0
        self.winner = False
        self._episode_ended = False

        self._states = self.game.board
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._states)

    def _legal_actions(self):
        return self.game.get_legal_moves(1)

    def _opponent_play(self):
        actions = self._legal_actions()
        return actions[0]

    def get_state(self):
        return copy.deepcopy(self._current_time_step)

    def set_state(self, time_step: TimeStep):
        self._current_time_step = time_step
        self._states = time_step.observation

    def _step(self, action):
        if self._current_time_step.is_last():
            return self._reset()

        player = 0

        action = int(action)
        move = action_to_move(action, BOARD_SIZE)

        move_result = self.game.move(player, move)

        if not move_result:
            return TimeStep(StepType.LAST, self.REWARD_ILLEGAL_MOVE, self._discount, self._states)

        is_final, reward = self._check_states()

        if is_final:
            return TimeStep(StepType.LAST, reward, self._discount,
                            self._states)

        opponent_action = self._opponent_play()
        self.game.move(1, opponent_action)

        is_final, reward = self._check_states()

        step_type = StepType.LAST if is_final else StepType.MID

        self.move += 1

        return TimeStep(step_type, reward, self._discount, self._states)

    def _check_states(self):
        if self.game.did_player_win(0):
            return True, self.REWARD_WIN
        if self.game.did_player_win(1):
            return True, self.REWARD_LOSS
        return False, 0

if __name__ == "__main__":
    env = ChineseCheckersEnv()
    env.reset()
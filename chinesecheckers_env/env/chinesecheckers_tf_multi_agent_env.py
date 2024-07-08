import random

from chinesecheckers_tf_env import ChineseCheckersEnv, BOARD_SIZE, MAX_MOVES
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType, TimeStep
from chinesecheckers_utils import action_to_move, get_legal_move_mask, move_to_action
import numpy as np
from chinesecheckers_game import show_frame

class ChineseCheckersMultiAgentEnv(ChineseCheckersEnv):

    def action_spec(self):
        action_spec = BoundedArraySpec((BOARD_SIZE ** 4,), np.int32, minimum=0, maximum=BOARD_SIZE ** 4 - 1)
        player_spec = BoundedArraySpec((1,), np.int32, minimum=0, maximum=1)
        return {
            'action': action_spec,
            'value': player_spec
        }

    def _step(self, action: np.ndarray):
        if self._current_time_step.is_last():
            return self._reset()

        player = action['player']

        action = int(action['action'])

        move = action_to_move(action, BOARD_SIZE)

        more_result = self.game.move(player, move)

        if not more_result:
            return TimeStep(StepType.LAST,
                            self.REWARD_ILLEGAL_MOVE,
                            self._discount,
                            self._states)

        is_final, reward = self._check_states()

        step_type = StepType.LAST if is_final else StepType.MID

        self.move += 1

        if self.move >= MAX_MOVES:
            return TimeStep(StepType.LAST, self.REWARD_LOSS, self._discount, self._states)

        return TimeStep(step_type, reward, self._discount, self._states)

if __name__ == "__main__":
    env = ChineseCheckersMultiAgentEnv()

    ts = env.reset()
    print(f"Reward: {ts.reward}")
    show_frame(env.game)

    random.seed(1)
    player = 0

    for i in range(10):
        legal_actions = env.game.get_legal_moves(player)
        legal_actions = list(map(move_to_action, legal_actions))


        action = {
            'action': np.asarray(random.choice(legal_actions)),
            'player': player
        }

        ts = env.step(action)
        print(f"Palyer: {player}, Action: {action_to_move(int(action['action']), BOARD_SIZE)}, Reward: {ts.reward}")
        show_frame(env.game)

        player = (1 + player) % 2
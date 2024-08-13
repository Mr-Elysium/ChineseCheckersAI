import random

BOARD_SIZE = 9

class Opponent:
    def __init__(self, env, difficulty):
        self.env = env.env
        self.difficulty = difficulty
        if self.difficulty == "random":
            self.get_action = self.random_opponent
        self.num_actions = BOARD_SIZE ** 4

    def random_opponent(self, action_mask):
        action = random.choices(list(range(self.num_actions)), action_mask)[0]
        return action


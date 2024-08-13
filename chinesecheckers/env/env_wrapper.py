from chinesecheckers.chinesecheckers_game import Move

class WrapperEnv:
    def __init__(self, env, lesson):
        self.env = env
        self.lesson = lesson

    def reward(self, done: bool, player: int, move: Move):
        if done:
            reward = self.lesson["rewards"]["win"]
        else:
            if move.move_to in self.env.game.get_home_coordinates(player):
                reward = self.lesson["rewards"]["home_jump"]
            else:
                reward = self.lesson["rewards"]["play_continues"]
            if player == 0:
                reward += max(((move.move_to.x + move.move_to.y) - (move.move_from.x + move.move_from.y)) * self.lesson["rewards"]["forward_jump"], 0)
            elif player == 1:
                reward += max(((move.move_from.x + move.move_from.y) - (move.move_to.x + move.move_to.y)) * self.lesson["rewards"]["forward_jump"], 0)
        return reward

    def last(self):
        return self.env.last()

    def step(self, action):
        self.env.step(action)

    def reset(self):
        self.env.reset()



from PIL import Image
import numpy as np
import pygame
from enum import IntEnum

class Direction(IntEnum):
    Right = 0
    UpRight = 1
    UpLeft = 2
    Left = 3
    DownLeft = 4
    DownRight = 5


class Position:
    direction_map = {
        Direction.Right: (+1, -1),
        Direction.UpRight: (+1, 0),
        Direction.UpLeft: (0, +1),
        Direction.Left: (-1, +1),
        Direction.DownLeft: (-1, 0),
        Direction.DownRight: (0, -1)
    }

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def neighbor(self, direction: Direction, distance: int = 1):
        x_delta, y_delta = Position.direction_map[direction]
        return Position(self.x + x_delta * distance, self.y + y_delta * distance)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return f"({self.x}, {self.y})"

class Move:
    def __init__(self, move_from: Position, move_to: Position):
        self.move_from = move_from
        self.move_to = move_to

    def __eq__(self, other):
        return self.move_from == other.move_from and self.move_to == other.move_to

    def __repr__(self):
        return f"({self.move_from.x}, {self.move_from.y}) --> ({self.move_to.x}, {self.move_to.y})"


class ChineseCheckers:
    colors = {
        -1: (154, 132, 73),
        0: (255, 0, 0),
        1: (0, 0, 255)
    }

    EMPTY_SPACE = -1

    def __init__(self, num_pins: int, render_mode: str = "human"):
        self.render_mode = render_mode

        self.triangle_size = int(np.floor(np.sqrt(num_pins * 2)))
        self.NUM_PINS = num_pins
        self.BOARD_SIZE = 2 * self.triangle_size + 1
        self.NUM_PLAYERS = 2

        self.clock = None

        self.init_game()

        self.window_size = 1000
        self.window = None

    def init_game(self):
        self._legal_moves = None
        self._game_over = False
        self.winner = {0: False, 1: False}

        self.board = np.full((self.BOARD_SIZE, self.BOARD_SIZE), self.EMPTY_SPACE)

        # Create Player 0 triangle
        for x in range(self.triangle_size):
            for y in range(self.triangle_size - x):
                self.board[x][y] = 0

        # Create Player 1 triangle
        for x in range(self.triangle_size):
            for y in range(self.triangle_size - x):
                self.board[self.BOARD_SIZE - x - 1][self.BOARD_SIZE - y - 1] = 1

    def find_legal_moves(self, player: int):
        moves = []

        for x in range(self.BOARD_SIZE):
            for y in range(self.BOARD_SIZE):
                if self._is_player(Position(x, y), player):
                    moves.extend(self.find_legal_moves_pin(Position(x, y)))

        if len(moves) == 0:
            self._game_over = True
            for agent in range(self.NUM_PLAYERS):
                if (agent != player):
                    self.winner[agent] = True

        self._legal_moves = moves

    def get_legal_moves(self, player: int):
        self.find_legal_moves(player)
        return self._legal_moves

    def find_legal_moves_pin(self, pos: Position):
        moves = []
        for direction in Direction:
            if self._in_bounds(pos.neighbor(direction)) and not self._is_occupied(pos.neighbor(direction)):
                moves.append(Move(pos, pos.neighbor(direction)))
        moves = self._get_jump_moves(moves, pos, pos)
        return moves

    def _get_jump_moves(self, moves, original_pos: Position, jump_pos: Position):
        for direction in Direction:
            if self._in_bounds(jump_pos.neighbor(direction)) and self._is_occupied(jump_pos.neighbor(direction)):
                if self._in_bounds(jump_pos.neighbor(direction, 2)) and not self._is_occupied(jump_pos.neighbor(direction, 2)) \
                        and jump_pos.neighbor(direction, 2) != original_pos:

                    move = Move(original_pos, jump_pos.neighbor(direction, 2))
                    if move not in moves:
                        moves.append(move)
                        moves = self._get_jump_moves(moves, original_pos, jump_pos.neighbor(direction, 2))

        return moves

    def _in_bounds(self, pos: Position):
        return 0 <= pos.x < self.BOARD_SIZE and 0 <= pos.y < self.BOARD_SIZE

    def _is_occupied(self, pos: Position):
        return self.board[pos.x][pos.y] != self.EMPTY_SPACE

    def _is_player(self, pos: Position, player: int):
        return self.board[pos.x][pos.y] == player

    def get_home_coordinates(self, player: int):
        home = []
        if player == 0:
            for x in range(self.triangle_size):
                for y in range(self.triangle_size - x):
                    home.append(Position(self.BOARD_SIZE - x - 1, self.BOARD_SIZE - y - 1))

        if player == 1:
            for x in range(self.triangle_size):
                for y in range(self.triangle_size - x):
                    home.append(Position(x, y))

        return home

    def get_player_coordinates(self, player: int):
        coords = []
        for x in range(self.BOARD_SIZE):
            for y in range(self.BOARD_SIZE):
                if self._is_player(Position(x, y), player):
                    coords.append(Position(x, y))

        assert len(coords) == self.NUM_PINS
        return coords

    def move(self, player: int, move: Move):
        if not self.is_move_legal(move, player):
            return None

        self._set_coordinate(move.move_from, self.EMPTY_SPACE)
        self._set_coordinate(move.move_to, player)

        if self.did_player_win(player):
            self.winner[player] = True
            self._game_over = True

    def is_move_legal(self, move: Move, player: int):
        self.find_legal_moves(player)
        return move in self._legal_moves

    def _set_coordinate(self, pos: Position, value: int):
        print(value, pos)
        self.board[pos.x][pos.y] = value

    def _get_coordinate(self, pos: Position):
        return self.board[pos.x][pos.y]

    def did_player_win(self, player: int):
        home = self.get_home_coordinates(player)
        for coord in home:
            if self._get_coordinate(coord) != player:
                return False

        return True

    def is_game_over(self):
        return self._game_over

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((241, 212, 133))

        def coordinate_to_pixel(pos: Position):
            size = 30
            screen_center_x, screen_center_y = self.window_size / 2, self.window_size / 2
            screen_origin_x, screen_origin_y = screen_center_x, screen_center_y - 12 * size
            return screen_origin_x + (pos.x - pos.y) * size * np.sqrt(3) * 0.5, screen_origin_y + (pos.x + pos.y) * size * 1.5

        pygame.init()
        for x in range(self.BOARD_SIZE):
            for y in range(self.BOARD_SIZE):
                pixel_x, pixel_y = coordinate_to_pixel(Position(x, y))
                value = self._get_coordinate(Position(x, y))
                pygame.draw.circle(
                    canvas,
                    self.colors[value],
                    (pixel_x, pixel_y),
                    10
                )

                font = pygame.font.SysFont("arial", 15)
                text = font.render(f"({x}, {y})", True, "black")
                canvas.blit(text, (pixel_x, pixel_y - 20))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(12)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

def show_frame(game):
    frame = game.render()
    Image.fromarray(frame, "RGB").show()


if __name__ == "__main__":
    game = ChineseCheckers(10, "rgba_array")
    show_frame(game)

    running = True
    while running:
        player = 0
        game.find_legal_moves(player)
        player_pins = game.get_player_coordinates(player)
        print(*player_pins, sep=", ")
        pin = input("choose pin to move 'x y'").split(' ')
        pin_x, pin_y = int(pin[0]), int(pin[1])

        possible_moves = game.find_legal_moves_pin(Position(pin_x, pin_y))
        print(*possible_moves, sep=", ")
        move = input("choose where to move 'x y'").split(' ')
        move_x, move_y = int(move[0]), int(move[1])

        game.move(player, Move(Position(pin_x, pin_y), Position(move_x, move_y)))
        show_frame(game)
        if game.did_player_win(player):
            running = False
            break

        player = 1
        game.find_legal_moves(player)
        player_pins = game.get_player_coordinates(player)
        print(*player_pins, sep=", ")
        pin = input("choose pin to move 'x y'").split(' ')
        pin_x, pin_y = int(pin[0]), int(pin[1])

        possible_moves = game.find_legal_moves_pin(Position(pin_x, pin_y))
        print(*possible_moves, sep=", ")
        move = input("choose where to move 'x y'").split(' ')
        move_x, move_y = int(move[0]), int(move[1])

        game.move(player, Move(Position(pin_x, pin_y), Position(move_x, move_y)))
        show_frame(game)
        if game.did_player_win(player):
            running = False
            break


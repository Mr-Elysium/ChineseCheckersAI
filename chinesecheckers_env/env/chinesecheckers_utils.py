import numpy as np

from chinesecheckers_game import Move, Position, ChineseCheckers
def action_to_move(action: int, board_size: int):
    index_from_board = action // (board_size ** 2)
    index_to_board = action % (board_size ** 2)
    move = Move(Position(index_from_board // board_size, index_from_board % board_size),
                Position(index_to_board // board_size, index_to_board % board_size))
    return move

def move_to_action(move: Move, board_size: int = 9):
    return (move.move_to.x * board_size + move.move_to.y) + (move.move_from.x * board_size + move.move_from.y) * (board_size ** 2)

def get_legal_move_mask(game: ChineseCheckers, player: int, board_size: int):
    mask = np.zeros((board_size ** 4)).flatten()
    for move in game.get_legal_moves(player):
        mask[move_to_action(move, board_size)] = 1
    return mask
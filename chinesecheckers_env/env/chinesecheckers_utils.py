import numpy as np
def check_winner(board):
    if board[0, -4:] == [1, 1, 1, 1] and board[0, -3:] == [1, 1, 1] and board[0, -2:] == [1, 1] and board[0, -1:] == [1]: return 1
    if board[-1, :4] == [2, 2, 2, 2] and board[-1, :3] == [2, 2, 2] and board[-1, :2] == [2, 2] and board[0, :1] == [2]: return 2
    return 0
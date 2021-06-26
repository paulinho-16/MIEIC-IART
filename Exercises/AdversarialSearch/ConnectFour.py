"""
alínea a):
    - State Representation:
        The state of this problem can be represented by a matrix representing the board, B[6,7] in this case, or B[N,M] in the general case.
        It will be filled with the values 0..2, where 0 represents an empty square, 1 represents a piece from player 1, and 2 represents a piece from player 2.
        It's also important to know the player to move (game turn): Pla, whose values are 1..2.
        For efficiency purposes, we could store the pair (Xl, Yl), which is the last square played.
    - Initial State:
        The initial state consists of the matrix B filled with zeros (0), and Pla = 1, since player 1 is the first to move.
    - Operators:
        exec_move(int Col)

        For playing at column Col, the top position of the column must be empty.
        In the effects, i represents the row with the highest value (lowest in the board) so that B[i][Col] was 0 before the move, this means that cell was the first empty cell of that column starting of the end of the board.

        ------------------------------------------------------------------------------
        |           Name           |  Preconditions   |            Effects           |
        ------------------------------------------------------------------------------
        |    exec_move(int Col)    |  B[0][Col] == 0  | B[i][Col] = Pla, Pla = 3-Pla |
        ------------------------------------------------------------------------------

    - Objective Test:
        The goal is reached when the matrix B contains a line of 4 pieces of the same colour/symbol horizontally, vertically, or diagonally, and the winner is the player with that colour/symbol.
        If the 42 pieces are played without any player getting a line, the final result will be a draw.
        The game state will be stored in the variable "winner", being: 0 - draw, 1 - win for player 1, 2 - win for player 2, -1 - game not finished.

alínea f):
    The evaluation function evalF4 is the one that obtains the best results. A larger depth for the algorithm, means a better prediction of good results done by minimax. However, a larger depth also means a bigger execution time of the algorithm.

alínea g)
    It could be also used the sequences of 2 pieces of a given player: a state with a sequence of 2 pieces of that player is better than a state without any sequence of more than 2 pieces of that player.
"""

# alíneas b), c), d), e)

import math
import random

NUM_ROWS = 6
NUM_COLS = 7

class State():
    def __init__(self, board: list, turn: int):
        self.board = board
        self.turn = turn
        self.num_moves = 0
    
    def __str__(self):
        result = "\n"
        for num in range(1, NUM_COLS + 1):
            result += "  " + str(num) if num == 1 else "   " + str(num)
        result += "\n"
        for row in self.board:
            result += "| "
            for column in row:
                result += str(column) + " | "
            if row != NUM_ROWS - 1:
                result += "\n"
        return result

    def copy(self):
        board_copy = []
        for row in self.board:
            row_copy = []
            for col in row:
                row_copy.append(col)
            board_copy.append(row_copy)
        turn = self.turn
        return State(board_copy, turn)

    def __eq__(self, other):
        return self.board == other.board and self.turn == other.turn

def initial_state():
    state = State([], 1)
    state.board = [[0 for j in range(NUM_COLS)] for i in range(NUM_ROWS)]
    return state

def change_turn(state):
    state.turn = 3 - state.turn

def valid_move(state, col):
    if not 1 <= col <= NUM_COLS:
        return False
    return state.board[0][col-1] == 0

def exec_move(state, col):
    if not valid_move(state, col):
        print("Invalid move, column must be between 1 and {}, in a column that is not full".format(NUM_COLS))
        return False
    i = NUM_ROWS - 1
    while state.board[i][col-1] != 0:
        i -= 1
    state.board[i][col-1] = state.turn
    change_turn(state)
    state.num_moves += 1
    return True

def count4(turn, p1, p2, p3, p4):
    return ((p1 == turn) + (p2 == turn) + (p3 == turn) + (p4 == turn)) == 4

def nlines4(state, turn):
    num = 0
    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            if j < NUM_COLS-3 and count4(turn, state.board[i][j], state.board[i][j+1], state.board[i][j+2], state.board[i][j+3]):
                num += 1
            if i < NUM_ROWS-3 and count4(turn, state.board[i][j], state.board[i+1][j], state.board[i+2][j], state.board[i+3][j]):
                num += 1
            if j < NUM_COLS-3 and i < NUM_ROWS-3 and count4(turn, state.board[i][j], state.board[i+1][j+1], state.board[i+2][j+2], state.board[i+3][j+3]):
                num += 1
            if j > 3 and i < NUM_ROWS-3 and count4(turn, state.board[i][j], state.board[i+1][j-1], state.board[i+2][j-2], state.board[i+3][j-3]):
                num += 1
    return num

def count3(turn, p1, p2, p3, p4):
    pec = (p1 == turn) + (p2 == turn) + (p3 == turn) + (p4 == turn)
    vaz = (p1 == 0) + (p2 == 0) + (p3 == 0) + (p4 == 0)
    return pec == 3 and vaz == 1

def nlines3(state):
    num = 0
    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            if j < NUM_COLS-3 and count3(state.turn, state.board[i][j], state.board[i][j+1], state.board[i][j+2], state.board[i][j+3]):
                num += 1
            if i < NUM_ROWS-3 and count3(state.turn, state.board[i][j], state.board[i+1][j], state.board[i+2][j], state.board[i+3][j]):
                num += 1
            if j < NUM_COLS-3 and i < NUM_ROWS-3 and count3(state.turn, state.board[i][j], state.board[i+1][j+1], state.board[i+2][j+2], state.board[i+3][j+3]):
                num += 1
            if j > 3 and i < NUM_ROWS-3 and count3(state.turn, state.board[i][j], state.board[i+1][j-1], state.board[i+2][j-2], state.board[i+3][j-3]):
                num += 1
    return num

def central(state, player):
    value = 0
    for col in [2,3,4]:
        for i in range(NUM_ROWS):
            if state.board[i][col] == player:
                value += 2 if col == 3 else 1
    return value

def evalF1(state, player):
    return nlines4(state, player) - nlines4(state, 3 - player)

def evalF2(state, player):
    state_copy = state.copy()
    state_copy.turn = player
    state_copy2 = state.copy()
    state_copy2.turn = 3 - player
    return 100*evalF1(state, player) + nlines3(state_copy) - nlines3(state_copy2)

def evalF3(state, player):
    return 100*evalF1(state, player) + central(state, player) - central(state, 3-player)

def evalF4(state, player):
    return 5*evalF2(state, player) + evalF3(state, player)

def game_over(state):
    if nlines4(state, 1) > 0:
        return 1
    if nlines4(state, 2) > 0:
        return 2
    if state.num_moves == NUM_ROWS*NUM_COLS:
        return 0
    return -1

def get_valid_locations(state):
    valid_locations = []
    for col in range(NUM_COLS):
        if valid_move(state, col):
            valid_locations.append(col)
    return valid_locations

def is_terminal_node(state):
    return nlines4(state, 1) > 0 or nlines4(state, 2) > 0 or len(get_valid_locations(state)) == 0

def minimax_alpha_beta(state, depth, alpha, beta, eval):
    valid_locations = get_valid_locations(state)
    is_terminal = is_terminal_node(state)
    if is_terminal:
        if nlines4(state, 1) > 0:
            return (None, -100000000000000)
        elif nlines4(state, 2) > 0:
            return (None, 100000000000000)
        else:
            return (None, 0)
    if depth == 0:
        return (None, eval(state, 2))
    if state.turn == 2:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            state_copy = state.copy()
            exec_move(state_copy, col)
            new_score = minimax_alpha_beta(state_copy, depth-1, alpha, beta, eval)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return (column, value)
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            state_copy = state.copy()
            exec_move(state_copy, col)
            new_score = minimax_alpha_beta(state_copy, depth-1, alpha, beta, eval)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return (column, value)

if __name__ == "__main__":
    state = initial_state()
    while True:
        print(state)
        while(True):
            if state.turn == 1:
                col = input("Player {} turn: ".format(state.turn))
            elif state.turn == 2:
                print("Player {} turn: ".format(state.turn))
                col, minimax_score = minimax_alpha_beta(state, 5, -math.inf, math.inf, evalF4)
            if exec_move(state, int(col)):
                break
        result = game_over(state)
        if result >= 0:
            break
    print(state)
    if result == 0:
        print("DRAW\n")
    if result == 1:
        print("PLAYER 1 WINS\n")
    if result == 2:
        print("PLAYER 2 WINS\n")
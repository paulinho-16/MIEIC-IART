"""
alínea a):
    - State Representation:
        The state of this problem can be represented by a matrix representing the board, B[3,3] in the case of the 9Puzzle example and B[4,4] in the case of the 16Puzzle example.
        In the general case, it will be a matrix B[N,N]. It will be filled with the values 0..8 for 9Puzzle, 0..15 for 16Puzzle and 0..N*N-1 for the general case, where 0 represents the empty square.
        For efficiency purposes, we could store the pair (Xs, Ys), which is the position of the empty square.
    - Initial State:
        The initial state consists of the matrix B filled with the desired initial state (might be a random distribution of the values), and the initial position of the empty square (Xi, Yi).
    - Operators:
        4 possibilities:
            up, down, left, right
            move(Dir)
            move(Xdir, Ydir)
            move(x1, y1, x2, y2)

        For moddeling 1:
        ---------------------------------------------------------------------------------------
        |   Name   |  Preconditions   |                     Effects                    | Cost |
        ---------------------------------------------------------------------------------------
        |    up    |      Ys > 0      | B[Xs,Ys] = B[Xs,Ys-1], B[Xs,Ys-1] = 0, Ys=Ys-1 |   1  |
        |   down   |     Ys < N-1     | B[Xs,Ys] = B[Xs,Ys+1], B[Xs,Ys+1] = 0, Ys=Ys+1 |   1  |
        |   left   |      Xs > 0      | B[Xs,Ys] = B[Xs-1,Ys], B[Xs-1,Ys] = 0, Xs=Xs-1 |   1  |
        |   right  |     Xs < N-1     | B[Xs,Ys] = B[Xs+1,Ys], B[Xs+1,Ys] = 0, Xs=Xs+1 |   1  |
        ---------------------------------------------------------------------------------------

    - Objective Test
        The goal is reached when the matrix B is ordered by value, from top to bottom, left to right, and the last square is the empty square (coords [N-1,N-1]).
"""

import sys
import heapq as hq
from datetime import datetime

class State():
    def __init__(self, board: list, empty_cell: tuple):
        self.board = board
        self.empty_cell = empty_cell
    
    def __str__(self):
        result = ""
        for row in self.board:
            result += str(row) + "\n"
        return result

    def copy(self):
        board_copy = []
        for row in self.board:
            row_copy = []
            for col in row:
                row_copy.append(col)
            board_copy.append(row_copy)
        (x, y) = self.empty_cell
        return State(board_copy, (x, y))

    def __eq__(self, other):
        return self.board == other.board and self.empty_cell == other.empty_cell

def up(state):
    (x, y) = state.empty_cell
    if x == 0:
        return False
    state.board[x][y] = state.board[x-1][y]
    state.board[x-1][y] = 0
    state.empty_cell = (x-1, y)
    return True

def down(state):
    (x, y) = state.empty_cell
    if x == len(state.board)-1:
        return False
    state.board[x][y] = state.board[x+1][y]
    state.board[x+1][y] = 0
    state.empty_cell = (x+1, y)
    return True

def left(state):
    (x, y) = state.empty_cell
    if y == 0:
        return False
    state.board[x][y] = state.board[x][y-1]
    state.board[x][y-1] = 0
    state.empty_cell = (x, y-1)
    return True

def right(state):
    (x, y) = state.empty_cell
    if y == len(state.board[0])-1:
        return False
    state.board[x][y] = state.board[x][y+1]
    state.board[x][y+1] = 0
    state.empty_cell = (x, y+1)
    return True

def objective(state):
    N = len(state.board)
    for i in range(0, N):
        for j in range(0, N):
            if state.board[i][j] != 0 and state.board[i][j] != i * N + j + 1:
                return False
    return True

def find_path(start, end):
    path = [end]
    operators = []
    while path[-1] != start:
        operators.append(path[-1].operator)
        path.append(path[-1].parent)
    path.reverse()
    operators.reverse()
    path_string = ""
    operators_string = ""
    for node in path:
        path_string += str(node) + "\n"
    for operator in operators:
        operators_string += operator.__name__ + " - "
    return (path_string[:-2], operators_string[:-2])

# alínea b)
def ucs(initial_state, operators, objective):
    visited = []
    entry_count = 1 # tie breaker between equal cost, so that states are stored according to insertion order
    queue = [(0, entry_count, initial_state)]
    hq.heapify(queue)
    while queue:
        elem = hq.heappop(queue)
        state = elem[2]
        if objective(state):
            return find_path(initial_state, state)
        if state not in visited:
            visited.append(state)
            neighbours = []
            for op in operators:
                new_state = state.copy()
                if op(new_state) and new_state != state and new_state != state.parent:
                    new_state.operator = op
                    neighbours.append(new_state)
            for neighbour in neighbours:
                neighbour.parent = state
                prev_cost = elem[0]
                cost = get_operator_cost(neighbour.operator)
                entry_count += 1
                hq.heappush(queue, (prev_cost + cost, entry_count, neighbour))

def get_operator_cost(op):
    if op == None: return 0
    elif op.__name__ == "up": return 1
    elif op.__name__ == "down": return 1
    elif op.__name__ == "left": return 1
    elif op.__name__ == "right": return 1

# alínea c)

def heuristic1(state): # number of incorrected placed pieces
    h1 = 0
    N = len(state.board)
    for i in range(N):
        for j in range(N):
            if state.board[i][j] != 0 and state.board[i][j] != i * N + j + 1:
                h1 += 1
    return h1

def heuristic2(state): # sum of Manhattan distances from incorrected placed pieces to their correct places
    h2 = 0
    N = len(state.board)
    for i in range(N):
        for j in range(N):
            if state.board[i][j] != 0 and state.board[i][j] != i * N + j + 1:
                xCorr = (state.board[i][j] - 1) // N
                yCorr = (state.board[i][j] - 1) % N
                h2 += abs(i-xCorr) + abs(j-yCorr)
    return h2

# alínea c1)
def gsa(initial_state, operators, objective):
    visited = []
    entry_count = 1 # tie breaker between equal heuristic values, so that states are stored according to insertion order
    queue = [(heuristic1(initial_state) + heuristic2(initial_state), entry_count, initial_state)]
    hq.heapify(queue)
    while queue:
        state = hq.heappop(queue)[2]
        if objective(state):
            return find_path(initial_state, state)
        if state not in visited:
            visited.append(state)
            neighbours = []
            for op in operators:
                new_state = state.copy()
                if op(new_state) and new_state != state and new_state != state.parent:
                    new_state.operator = op
                    neighbours.append(new_state)
            for neighbour in neighbours:
                neighbour.parent = state
                heuristic = heuristic1(neighbour) + heuristic2(neighbour)
                entry_count += 1
                hq.heappush(queue, (heuristic, entry_count, neighbour))

# alínea c2)
def asa(initial_state, operators, objective):
    visited = []
    entry_count = 1 # tie breaker between equal heuristic values, so that states are stored according to insertion order
    queue = [(heuristic1(initial_state) + heuristic2(initial_state), entry_count, initial_state)]
    hq.heapify(queue)
    while queue:
        elem = hq.heappop(queue)
        state = elem[2]
        if objective(state):
            return find_path(initial_state, state)
        if state not in visited:
            visited.append(state)
            neighbours = []
            for op in operators:
                new_state = state.copy()
                if op(new_state) and new_state != state and new_state != state.parent:
                    new_state.operator = op
                    neighbours.append(new_state)
            for neighbour in neighbours:
                neighbour.parent = state
                heuristic = heuristic1(neighbour) + heuristic2(neighbour)
                prev_cost = elem[0] - (heuristic1(state) + heuristic2(state))
                cost = get_operator_cost(neighbour.operator)
                entry_count += 1
                hq.heappush(queue, (prev_cost + cost + heuristic, entry_count, neighbour))

# alínea d)
"""
    Uniform Cost Search takes a lot more time than the others, and A* is definitely the best algorithm, since it takes less time and it is optimal.
"""
def compare_times(operators):
    prob1 = [
        [1, 2, 3],
        [5, 0, 6],
        [4, 7, 8]
    ]
    initial_state = State(prob1, (1, 1))
    initial_state.parent = State([], (-1, -1))
    initial_state.operator = None

    print("--- Prob1 ---")

    start = datetime.now()
    res = ucs(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob1 by Uniform Cost Search:\n' + res[1])
    print("Uniform Cost Search took " + str(time) + " microseconds to solve prob1")

    start = datetime.now()
    res = gsa(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob1 by Greedy Search algorithm:\n' + res[1])
    print("Greedy Search algorithm took " + str(time) + " microseconds to solve prob1")

    start = datetime.now()
    res = asa(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob1 by A* algorithm:\n' + res[1])
    print("A* algorithm took " + str(time) + " microseconds to solve prob1")

    prob2 = [
        [1, 3, 6],
        [5, 2, 0],
        [4, 7, 8]
    ]
    initial_state = State(prob2, (1, 2))
    initial_state.parent = State([], (-1, -1))
    initial_state.operator = None

    print("\n--- Prob2 ---")

    start = datetime.now()
    res = ucs(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob2 by Uniform Cost Search:\n' + res[1])
    print("Uniform Cost Search took " + str(time) + " microseconds to solve prob2")

    start = datetime.now()
    res = gsa(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob2 by Greedy Search algorithm:\n' + res[1])
    print("Greedy Search algorithm took " + str(time) + " microseconds to solve prob2")

    start = datetime.now()
    res = asa(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob2 by A* algorithm:\n' + res[1])
    print("A* algorithm took " + str(time) + " microseconds to solve prob2")

    prob3 = [
        [1, 6, 2],
        [5, 7, 3],
        [0, 4, 8]
    ]
    initial_state = State(prob3, (2, 0))
    initial_state.parent = State([], (-1, -1))
    initial_state.operator = None

    print("\n--- Prob3 ---")

    start = datetime.now()
    res = ucs(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob3 by Uniform Cost Search:\n' + res[1])
    print("Uniform Cost Search took " + str(time) + " microseconds to solve prob3")

    start = datetime.now()
    res = gsa(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob3 by Greedy Search algorithm:\n' + res[1])
    print("Greedy Search algorithm took " + str(time) + " microseconds to solve prob3")

    start = datetime.now()
    res = asa(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob3 by A* algorithm:\n' + res[1])
    print("A* algorithm took " + str(time) + " microseconds to solve prob3")

    prob4 = [
        [5, 1, 3, 4],
        [2, 0, 7, 8],
        [10, 6, 11, 12],
        [9, 13, 14, 15]
    ]
    initial_state = State(prob4, (1, 1))
    initial_state.parent = State([], (-1, -1))
    initial_state.operator = None

    print("\n--- Prob4 ---")

    start = datetime.now()
    res = ucs(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob4 by Uniform Cost Search:\n' + res[1])
    print("Uniform Cost Search took " + str(time) + " microseconds to solve prob4")

    start = datetime.now()
    res = gsa(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob4 by Greedy Search algorithm:\n' + res[1])
    print("Greedy Search algorithm took " + str(time) + " microseconds to solve prob4")

    start = datetime.now()
    res = asa(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution of prob4 by A* algorithm:\n' + res[1])
    print("A* algorithm took " + str(time) + " microseconds to solve prob4")

if __name__ == "__main__":
    initial_board = [
        [1, 2, 3],
        [4, 6, 0],
        [7, 5, 8]
    ]
    initial_state = State(initial_board, (1, 2))
    initial_state.parent = State([], (-1, -1))
    initial_state.operator = None
    operators = [up, down, left, right]
    if len(sys.argv) != 2:
        print('Usage: NPuzzleProblem.py <algorithm>')
        print('\t<algorithm> -> informed search algorithm (ucs, gsa, asa)')
        exit(1)
    if (sys.argv[1].lower() == 'ucs'):
        print('--- Uniform Cost Search (UCS) ---')
        res = ucs(initial_state, operators, objective)
    elif (sys.argv[1].lower() == 'gsa'):
        print('--- Greedy Search Algorithm (GSA) ---')
        res = gsa(initial_state, operators, objective)
    elif (sys.argv[1].lower() == 'asa'):
        print('--- A* algorithm (ASA) ---')
        res = asa(initial_state, operators, objective)
    else:
        print('Invalid algorithm. Available: ucs, gsa, asa')
        exit(2)
    if not res:
        print('No solution found')
        exit(3)
    print('\nStates:\n' + res[0])
    print('\nOperators:\n' + res[1])

    print('\nComparing times of the algorithms:\n')
    compare_times(operators)
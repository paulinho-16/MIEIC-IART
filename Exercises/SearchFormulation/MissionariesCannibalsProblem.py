"""
alínea a):
    - State Representation:
        The state of this problem can be represented by three integers, nm, nc and nb, where nm represents
        the number of missionaries on the initial margin, nc represents the number of cannibals on the initial margin,
        and nb represents the number of boats on the initial margin.
        nm must be between 0 and 3, nc must be between 0 and 3, and nb between 0 and 1 (0 <= nm <= 3, 0 <= nc <= 3, 0 <= nb <= 1).
    - Initial State:
        Since there are three missionaries and three cannibals on the initial margin, as well as the boat, the initial state is 3-3-1.
    - Operators:
        3 possibilities:
            MM1, CC1, MC1, M1, C1, MM2, CC2, MC2, M2, C2
            MM(Dir), CC(Dir), MC(Dir), M(Dir), C(Dir)
            Trip(NM, NC, Dir)

        For moddeling 1:
        (the number of cannibals has to be smaller than the number of missionaries only when the number of missionaries is greater than 0. It was ommited for readibility purposes)
        -------------------------------------------------------------------------------------------
        |   Name   |             Preconditions              |            Effects           | Cost |
        -------------------------------------------------------------------------------------------
        |    MM1   |    nm >= 2 /\ nb == 1 /\ nc <= nm-2    |      nm = nm-2, nb = 0       |   1  |
        |    CC1   |  nc >= 2 /\ nb == 1 /\ 3-nm >= 3-nc+2  |      nc = nc-2, nb = 0       |   1  |
        |    MC1   |      nm >= 1 /\ nc >= 1 /\ nb == 1     | nm = nm-1, nc = nc-1, nb = 0 |   1  |
        |    M1    |    nm >= 1 /\ nb == 1 /\ nm-1 >= nc    |      nm = nm-1, nb = 0       |   1  |
        |    C1    |  nc >= 1 /\ nb == 1 /\ 3-nm >= 3-nc+1  |      nc = nc-1, nb = 0       |   1  |
        |    MM2   | 3-nm >= 2 /\ nb == 0 /\ 3-nm-2 >= 3-nc |      nm = nm+2, nb = 1       |   1  |
        |    CC2   |   3-nc >= 2 /\ nb == 0 /\ nm >= nc+2   |      nc = nc+2, nb = 1       |   1  |
        |    MC2   |   3-nm >= 1 /\ 3-nc >= 1 /\ nb == 0    | nm = nm+1, nc = nc+1, nb = 1 |   1  |
        |    M2    | 3-nm >= 1 /\ nb == 0 /\ 3-nm-1 >= 3-nc |      nm = nm+1, nb = 1       |   1  |
        |    C2    |   3-nc >= 1 /\ nb == 0 /\ nm >= nc+1   |      nc = nc+1, nb = 1       |   1  |
        -------------------------------------------------------------------------------------------

    - Objective Test
        The goal is reached when all 6 people are on the other margin of the river, without ever leaving more cannibals than missionaries on one of the margins.
        Therefore, the final state is 0-0-0.
        
alínea b):
    One possible solution:
        CC1, C2, CC1, C2, MM1, MC2, MM1, C2, CC1, M2, MC1

                         3-3-1
              /            |          \
           3-1-0         2-2-0       3-2-0
             |
           3-2-1
          /    \
       3-0-0  3-1-0
         |
       3-1-1
         |
       1-1-0
         |
       2-2-1
         |
       0-2-0
         |
       0-3-1
         |
       0-1-0
         |
       1-1-1
         |
       0-0-0
"""

# alínea c):
import sys

N = 3

class State():
    def __init__(self, nm=N, nc=N, nb=1):
        [self.nm, self.nc, self.nb] = [nm, nc, nb]
    
    def write(self):
        return '[' + str(self.nm) + ', ' + str(self.nc) + ', ' + str(self.nb) + ']'

    def copy(self):
        return State(self.nm, self.nc, self.nb)

    def __eq__(self, other):
        return self.nm == other.nm and self.nc == other.nc and self.nb == other.nb

def mm1(state):
    if state.nb == 1 and state.nm >= 2 and ((state.nc <= state.nm - 2 and state.nm - 2 > 0) or state.nm - 2 == 0):
        state.nm -= 2
        state.nb = 0
        return True
    return False

def cc1(state):
    if state.nb == 1 and state.nc >= 2 and ((N-state.nm >= N-state.nc+2 and N-state.nm > 0) or N-state.nm == 0):
        state.nc -= 2
        state.nb = 0
        return True
    return False

def mc1(state):
    if state.nb == 1 and state.nm >= 1 and state.nc >= 1 and ((N-state.nm+1 >= N-state.nc+1 and N-state.nm+1 > 0) or N-state.nm+1 == 0):
        state.nm -= 1
        state.nc -= 1
        state.nb = 0
        return True
    return False

def m1(state):
    if state.nb == 1 and state.nm >= 1 and ((state.nm-1 >= state.nc and state.nm-1 > 0 and ((N-state.nm+1 >= N-state.nc and N-state.nm+1 > 0) or N-state.nm+1 == 0)) or state.nm-1 == 0):
        state.nm -= 1
        state.nb = 0
        return True
    return False

def c1(state):
    if state.nb == 1 and state.nc >= 1 and ((N-state.nm >= N-state.nc+1 and N-state.nm > 0) or N-state.nm == 0):
        state.nc -= 1
        state.nb = 0
        return True
    return False

def mm2(state):
    if state.nb == 0 and N-state.nm >= 2 and ((N-state.nm-2 >= N-state.nc and N-state.nm-2 > 0) or N-state.nm-2 == 0):
        state.nm += 2
        state.nb = 1
        return True
    return False

def cc2(state):
    if state.nb == 0 and N-state.nc >= 2 and ((state.nm >= state.nc+2 and state.nm > 0) or state.nm == 0):
        state.nc += 2
        state.nb = 1
        return True
    return False

def mc2(state):
    if state.nb == 0 and N-state.nm >= 1 and N-state.nc >= 1 and ((state.nm+1 >= state.nc+1 and state.nm+1 > 0) or state.nm+1 == 0):
        state.nm += 1
        state.nc += 1
        state.nb = 1
        return True
    return False

def m2(state):
    if state.nb == 0 and N-state.nm >= 1 and ((N-state.nm-1 >= N-state.nc and N-state.nm-1 > 0 and state.nm+1 >= state.nc) or N-state.nm-1 == 0):
        state.nm += 1
        state.nb = 1
        return True
    return False

def c2(state):
    if state.nb == 0 and N-state.nc >= 1 and ((state.nm >= state.nc+1 and state.nm > 0) or state.nm == 0):
        state.nc += 1
        state.nb = 1
        return True
    return False

def objective(state):
    return state.nm == 0 and state.nc == 0 and state.nb == 0

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
        path_string += node.write() + " - "
    for operator in operators:
        operators_string += operator.__name__ + " - "
    return (path_string[:-2], operators_string[:-2])

# alínea c1)
def bfs(initial_state, operators, objective):
    visited = []
    queue = [initial_state]
    while queue:
        state = queue.pop(0)
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
                queue.append(neighbour)
    return False

# alínea c2)
def dfs(start, objective, operators, limit=10, path=[], visited=[]):
    if len(path) >= limit:
        return []
    if start in visited:
        return path
    path += [start]
    visited += [start]

    if objective(start):
        return path

    neighbours = []
    for op in operators:
        new_state = start.copy()
        if op(new_state) and new_state != start and new_state != start.parent:
            new_state.operator = op
            neighbours.append(new_state)
    for neighbour in neighbours:
        neighbour.parent = start
    for nbr in neighbours:
        if nbr not in visited:
            return dfs(nbr, objective, operators, limit, path, visited)

# alínea c3)
def ids(start, objective, operators, limit=10):
    def ids_aux(path, objective, limit):
        current = path[-1]
        if objective(current):
            return path
        if limit <= 0:
            return None
        neighbours = []
        for op in operators:
            new_state = current.copy()
            if op(new_state) and new_state != current and new_state != current.parent:
                new_state.operator = op
                neighbours.append(new_state)
        for neighbour in neighbours:
            neighbour.parent = current
            new_path = list(path)
            new_path.append(neighbour)
            result = ids_aux(new_path, objective, limit-1)
            if result is not None:
                return result
        
    for depth in range(0, limit):
        result = ids_aux([start], objective, depth)
        if result is None:
            continue
        return result
    return []

if __name__ == "__main__":
    initial_state = State()
    initial_state.parent = State(-1, -1, -1)
    operators = [mm1, cc1, mc1, m1, c1, mm2, cc2, mc2, m2, c2]
    if len(sys.argv) != 2:
        print('Usage: MissionariesCannibalsProblem.py <algorithm>')
        print('\t<algorithm> -> search algorithm (bfs, dfs, ids)')
        exit(1)
    if (sys.argv[1].lower() == 'bfs'):
        print('--- Breadth-First Search (BFS) ---')
        res = bfs(initial_state, operators, objective)
    elif (sys.argv[1].lower() == 'dfs'):
        print('--- Depth-First Search with limited depth (DFS) ---')
        sol = dfs(initial_state, objective, operators, 12)
        res = find_path(initial_state, sol[-1]) if sol else []
    elif (sys.argv[1].lower() == 'ids'):
        print('--- Iterative deepening strategy (IDS) ---')
        sol = ids(initial_state, objective, operators, 12)
        res = find_path(initial_state, sol[-1]) if sol else []
    else:
        print('Invalid algorithm. Available: bfs, dfs, ids')
    if not res:
        print('No solution found')
        exit(2)
    print('States:\n' + res[0])
    print('Operators:\n' + res[1])
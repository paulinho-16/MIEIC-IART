"""
alínea a):
    - State Representation:
        The state of this problem can be represented by two integers, x and y, where x represents
        the number of liters of water that the bucket A (capacity of 4 liters) contains, and y
        represents the number of liters of water that the bucket B (capacity of 3 liters) contains.
        x must be between 0 and 4, and y between 0 and 3 (0 <= x <= 4, 0 <= y <= 3).
    - Initial State:
        Since both buckets are initially empty, the initial state is 0-0.
    - Operators:
        3 possibilities:
            Emp1, Emp2, Fill1, Fill2, Pour12a, Pour12b, Pour21a, Pour21b (a -> until second bucket is full, b -> until first bucket is empty)
            Emp1, Emp2, Fill1, Fill2, Pour12, Pour21
            Emp(x), Fill(x), Pour(x,y)

        For moddeling 1:
        ---------------------------------------------------------------------
        |   Name   |       Preconditions        |       Effects      | Cost |
        ---------------------------------------------------------------------
        |   Emp1   |            x > 0           |        x = 0       |   1  |
        |   Emp2   |            y > 0           |        y = 0       |   1  |
        |   Fill1  |            x < 4           |        x = 4       |   1  |
        |   Fill2  |            y < 3           |        y = 3       |   1  |
        |  Pour12a | x > 0 /\ y < 3 /\ x >= 3-y | x = x-(3-y), y = 3 |   1  |
        |  Pour12b | x > 0 /\ y < 3 /\ x < 3-y  |  x = 0, y = y + x  |   1  |
        |  Pour21a | y > 0 /\ x < 4 /\ y >= 4-x | y = y-(4-x), x = 4 |   1  |
        |  Pour21b | y > 0 /\ x < 4 /\ y < 4-x  |  y = 0, x = x + y  |   1  |
        ---------------------------------------------------------------------

    - Objective Test
        The goal is reached when the bucket A contains N liters, and in this case we're using N=2.
        As the number of liters that the bucket B contains is not important at the end, the final state is N-y, or in this case, 2-y.
        
alínea b):
    Two possible solutions:
        Fill1, Pour12a, Emp2, Pour12b, Fill1, Pour12a
        Fill2, Pour21b, Fill2, Pour21a, Emp1, Pour21b

               0-0
           /         \
         4-0          0-3
        /    \      /    \
      4-3   1-3   4-3    3-0
       |     |     |      |
      ...   1-0   4-0    3-3  
             |     |      |
            0-1   ...    4-2
             |            |
            4-1          0-2
             |            |
            2-3          2-0
"""

# alínea c):
import sys

C1 = 4; C2 = 3; N = 2

class State():
    def __init__(self, W1=0, W2=0):
        [self.W1, self.W2] = [W1, W2]
    
    def write(self):
        return '[' + str(self.W1) + ', ' + str(self.W2) + ']'

    def copy(self):
        return State(self.W1, self.W2)

    def __eq__(self, other):
        return self.W1 == other.W1 and self.W2 == other.W2

def emp1(state):
    if state.W1 > 0:
        state.W1 = 0
        return True
    return False

def emp2(state):
    if state.W2 > 0:
        state.W2 = 0
        return True
    return False

def fill1(state):
    if state.W1 < C1:
        state.W1 = C1
        return True
    return False

def fill2(state):
    if state.W2 < C2:
        state.W2 = C2
        return True
    return False

def pour12a(state):
    if state.W1 > 0 and state.W2 < C2 and state.W1 >= C2-state.W2:
        state.W1 -= C2 - state.W2
        state.W2 = C2
        return True
    return False

def pour12b(state):
    if state.W1 > 0 and state.W2 < C2 and state.W1 < C2-state.W2:
        state.W2 += state.W1
        state.W1 = 0
        return True
    return False

def pour21a(state):
    if state.W2 > 0 and state.W1 < C1 and state.W2 >= C1-state.W1:
        state.W2 -= C1 - state.W1
        state.W1 = C1
        return True
    return False

def pour21b(state):
    if state.W2 > 0 and state.W1 < C1 and state.W2 < C1-state.W1:
        state.W1 += state.W2
        state.W2 = 0
        return True
    return False

def objective(state):
    return state.W1 == N

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
    initial_state.parent = State(-1, -1)
    operators = [fill1, fill2, emp1, emp2, pour12a, pour12b, pour21a, pour21b]
    if len(sys.argv) != 2:
        print('Usage: TwoBucketsProblem.py <algorithm>')
        print('\t<algorithm> -> search algorithm (bfs, dfs, ids)')
        exit(1)
    if (sys.argv[1].lower() == 'bfs'):
        print('--- Breadth-First Search (BFS) ---')
        res = bfs(initial_state, operators, objective)
    elif (sys.argv[1].lower() == 'dfs'):
        print('--- Depth-First Search with limited depth (DFS) ---')
        sol = dfs(initial_state, objective, operators, 10)
        res = find_path(initial_state, sol[-1]) if sol else []
    elif (sys.argv[1].lower() == 'ids'):
        print('--- Iterative deepening strategy (IDS) ---')
        sol = ids(initial_state, objective, operators, 10)
        res = find_path(initial_state, sol[-1]) if sol else []
    else:
        print('Invalid algorithm. Available: bfs, dfs, ids')
    if not res:
        print('No solution found')
        exit(2)
    print('States:\n' + res[0])
    print('Operators:\n' + res[1])
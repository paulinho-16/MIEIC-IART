import sys
import heapq as hq
from datetime import datetime

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

# alínea a)

def get_operator_cost(op):
    [fill1, fill2, emp1, emp2, pour12a, pour12b, pour21a, pour21b]
    if op == None: return 0
    elif op.__name__ == "fill1": return 1
    elif op.__name__ == "fill2": return 1
    elif op.__name__ == "emp1": return 1
    elif op.__name__ == "emp2": return 1
    elif op.__name__ == "pour12a": return 1
    elif op.__name__ == "pour12b": return 1
    elif op.__name__ == "pour21a": return 1
    elif op.__name__ == "pour21b": return 1

def heuristic(state): # number of liters of water needed for filling the first bucket with the desired value, in this case 2 liters
    return abs(state.W1 - N)

# alínea a1)
def gsa(initial_state, operators, objective):
    visited = []
    entry_count = 1 # tie breaker between equal heuristic values, so that states are stored according to insertion order
    queue = [(heuristic(initial_state), entry_count, initial_state)]
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
                heuristic_value = heuristic(neighbour)
                entry_count += 1
                hq.heappush(queue, (heuristic_value, entry_count, neighbour))

# alínea a2)
def asa(initial_state, operators, objective):
    visited = []
    entry_count = 1 # tie breaker between equal heuristic values, so that states are stored according to insertion order
    queue = [(heuristic(initial_state), entry_count, initial_state)]
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
                heuristic_value = heuristic(neighbour)
                prev_cost = elem[0] - heuristic(state)
                cost = get_operator_cost(neighbour.operator)
                entry_count += 1
                hq.heappush(queue, (prev_cost + cost + heuristic_value, entry_count, neighbour))

# alínea d)
"""
    There is no significant difference between the execution times since the state space is too small
"""
def compare_times(operators):
    initial_state = State()
    initial_state.parent = State(-1, -1)

    start = datetime.now()
    res = bfs(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('Solution by Breadth-First Search algorithm:\n' + res[1])
    print("Breadth-First Search algorithm took " + str(time) + " microseconds")

    start = datetime.now()
    sol = dfs(initial_state, objective, operators, 10)
    res = find_path(initial_state, sol[-1]) if sol else []
    time = (datetime.now() - start).microseconds
    print('\nSolution by Depth-First Search with limited depth:\n' + res[1])
    print("Depth-First Search with limited depth took " + str(time) + " microseconds")

    start = datetime.now()
    sol = ids(initial_state, objective, operators, 10)
    res = find_path(initial_state, sol[-1]) if sol else []
    time = (datetime.now() - start).microseconds
    print('\nSolution by Iterative deepening strategy:\n' + res[1])
    print("Iterative deepening strategy took " + str(time) + " microseconds")

    start = datetime.now()
    res = gsa(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution by Greedy Search algorithm:\n' + res[1])
    print("Greedy Search algorithm took " + str(time) + " microseconds")

    start = datetime.now()
    res = asa(initial_state, operators, objective)
    time = (datetime.now() - start).microseconds
    print('\nSolution by A* algorithm:\n' + res[1])
    print("A* algorithm took " + str(time) + " microseconds")

if __name__ == "__main__":
    initial_state = State()
    initial_state.parent = State(-1, -1)
    operators = [fill1, fill2, emp1, emp2, pour12a, pour12b, pour21a, pour21b]
    if len(sys.argv) != 2:
        print('Usage: TwoBucketsProblem.py <algorithm>')
        print('\t<algorithm> -> informed search algorithm (gsa, asa)')
        exit(1)
    if (sys.argv[1].lower() == 'gsa'):
        print('--- Greedy Search Algorithm (GSA) ---')
        res = gsa(initial_state, operators, objective)
    elif (sys.argv[1].lower() == 'asa'):
        print('--- A* algorithm (ASA) ---')
        res = asa(initial_state, operators, objective)
    else:
        print('Invalid algorithm. Available: gsa, asa')
    if not res:
        print('No solution found')
        exit(2)
    print('States:\n' + res[0])
    print('Operators:\n' + res[1])

    print('\nComparing times of the algorithms:\n')
    compare_times(operators)
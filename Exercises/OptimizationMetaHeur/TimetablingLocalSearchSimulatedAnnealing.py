import random
from math import exp

class Problem:
    # alínea a)
    """ Solution Representation: Array with 12 positions/disciplines - solution[12] with values 1..4 (slots) """
    def __init__(self) -> None:
        with open('Exercises/OptimizationMetaHeur/ProblemInfo.txt', 'r') as f:
            (self.numSlots, self.numDisciplines, self.numStudents) = (int(n) for n in f.readline().split())
            self.classes = [[int(n) for n in f.readline().split()] for _ in range(self.numDisciplines)]
        self.solution = [random.randrange(1, self.numSlots + 1) for _ in range(self.numDisciplines)]
        # alínea f)
        #self.solution = [1,1,1,1,1,1,1,1,1,1,1,1]
        #self.solution = [1,1,1,2,2,2,3,3,3,4,4,1]
        #self.solution = [1,1,4,2,2,2,3,3,3,4,4,1]
        #self.solution = [1,2,3,4,1,2,3,4,1,2,3,4]

    # alínea b)
    def incompatibilities(self, d1: int, d2: int) -> int:
        return len(set(self.classes[d1]).intersection(set(self.classes[d2])))

    # alínea c)
    def evaluate(self, solution) -> int:
        score = 0
        for d1 in range(self.numDisciplines - 1):
            for d2 in range(d1 + 1, self.numDisciplines):
                if solution[d1] == solution[d2]:
                    score += self.incompatibilities(d1, d2)
        return score

    # alínea d)
    def neighbour1(self, solution):
        d1 = random.randrange(self.numDisciplines)
        new_slot = random.randrange(1, self.numSlots + 1)
        while new_slot == solution[d1]:
            new_slot = random.randrange(1, self.numSlots + 1)
        solution[d1] = new_slot
        return solution

    def neighbour2(self, solution):
        d1 = random.randrange(self.numDisciplines)
        d2 = random.randrange(self.numDisciplines)
        while d1 == d2 or solution[d1] == solution[d2]:
            d2 = random.randrange(self.numDisciplines)
        solution[d1], solution[d2] = solution[d2], solution[d1]
        return solution

    def neighbour3(self, solution):
        if random.choice([0, 1]) == 0: return self.neighbour1(solution)
        else: return self.neighbour2(solution)

    # alínea e)
    def hillClimbing(self):
        it = 0
        while it < 1000:
            neighbour = self.neighbour3(self.solution.copy())
            it += 1
            if self.evaluate(neighbour) < self.evaluate(self.solution):
                self.solution = neighbour
                it = 0
        return self.solution

    def simulatedAnnealing(self):
        it = 0
        T = 1000
        while it < 1000:
            T *= 0.9
            neighbour = self.neighbour3(self.solution.copy())
            it += 1
            delta = self.evaluate(self.solution) - self.evaluate(neighbour)
            if delta > 0 or exp(delta / T) > random.random():
                self.solution = neighbour
                it = 0
        return self.solution

if __name__ == "__main__":
    problem1 = Problem()
    problem2 = Problem()
    sol1 = problem1.hillClimbing()
    print(f"Hill Climbing solution: {sol1} - Incompatibilities: {problem1.evaluate(sol1)}")
    sol2 = problem2.simulatedAnnealing()
    print(f"Simulated Annealing solution: {sol2} - Incompatibilities: {problem2.evaluate(sol2)}")

"""
    alínea g):
        The problem is already starting with a random solution. To implement the "Random Restarts" method,
        we would need, given a small probability, to completely restart the problem, by ignoring the current
        best solution and generating and storing a new random solution.
    alínea h):
        Both Hill Climbing and Simulated Annealing algorithms provide good solutions (most of the times with 0 incompatibilities).
        However, Simulated Annealing allows us to escape local minimums, even though it takes more time to execute.
    alínea i):
        It's possible to create several instances of the problem, with different dimensions (varying the number of disciplines and slots)
        and difficulties (varying the students enrolled in the disciplines).
"""
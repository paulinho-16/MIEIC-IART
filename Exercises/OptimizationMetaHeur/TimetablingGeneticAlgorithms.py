import random

class Problem:
    # alínea a)
    """ Solution Representation: Array with 12 positions/disciplines - solution[12] with values 1..4 (slots) """
    def __init__(self) -> None:
        with open('Exercises/OptimizationMetaHeur/ProblemInfo.txt', 'r') as f:
            (self.numSlots, self.numDisciplines, self.numStudents) = (int(n) for n in f.readline().split())
            self.classes = [[int(n) for n in f.readline().split()] for _ in range(self.numDisciplines)]
        self.solution = [random.randrange(1, self.numSlots + 1) for _ in range(self.numDisciplines)]
        self.numIndividuals = 10

    def incompatibilities(self, d1: int, d2: int) -> int:
        return len(set(self.classes[d1]).intersection(set(self.classes[d2])))

    # alínea b)
    def evaluate(self, solution) -> int:
        score = 0
        for d1 in range(self.numDisciplines - 1):
            for d2 in range(d1 + 1, self.numDisciplines):
                if solution[d1] == solution[d2]:
                    score += self.incompatibilities(d1, d2)
        return 100 - score # evaluation maximization

    # alínea c)
    def crossover(self, ind1, ind2):
        separation = int(self.numDisciplines / 2)
        return ind1[:separation] + ind2[separation:]

    # alínea d)
    def selection(self):
        best_idx = 0
        worst_idx = self.numIndividuals - 1
        rnd_idx = random.randrange(self.numIndividuals)
        while rnd_idx == worst_idx:
            rnd_idx = random.randrange(self.numIndividuals)
        return (best_idx, worst_idx, rnd_idx)

    # alínea e)
    def mutation(self, solution):
        d1 = random.randrange(self.numDisciplines)
        new_slot = random.randrange(1, self.numSlots + 1)
        while new_slot == solution[d1]:
            new_slot = random.randrange(1, self.numSlots + 1)
        solution[d1] = new_slot
        return solution

    # alínea f)
    def geneticAlgorithm(self):
        self.generation = [[random.randrange(1, self.numSlots + 1) for _ in range(self.numDisciplines)] for _ in range(self.numIndividuals)]
        self.generation.sort(key=self.evaluate, reverse=True)
        it = 0
        while it != 1000:
            (best_idx, worst_idx, rnd_idx) = self.selection()
            new_ind = self.crossover(self.generation[best_idx], self.generation[rnd_idx])
            self.generation[worst_idx] = new_ind
            if random.random() < 0.05:
                mutated = self.mutation(self.generation[rnd_idx])
                self.generation[rnd_idx] = mutated
            self.generation.sort(key=self.evaluate, reverse=True)
            it += 1
        return self.generation[0]

if __name__ == "__main__":
    problem = Problem()
    sol = problem.geneticAlgorithm()
    print(f"Genetic Algorithm solution: {sol} - Incompatibilities: {100 - problem.evaluate(sol)}")

"""
    alínea g):
        It's possible to create several instances of the problem, with different dimensions (varying the number of disciplines and slots)
        and difficulties (varying the students enrolled in the disciplines).
    alínea h):
        The problem could be solved using different parameterizations of the algorithm, namely by varying the population size, number of generations,
        selection method, crossover method, mutation method, etc.
    alínea i):
        The Genetic Algorithm obtains worse results than the Hill Climbing and Simulated Annealing algorithms.
        However, it allows us to better explore the state space. It is also fast to execute.
"""
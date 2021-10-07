import random
import math
import heapq
from router import Router

class Strategy():
    def __init__(self, buildPlan):
        buildPlan.generate_routers()
        self.buildPlan = buildPlan
        self.score = buildPlan.get_total_score()
        print("INITIAL SCORE: ", self.score)

    def algorithm(self, iterations):
        pass

class HillClimbing(Strategy):

    def algorithm(self, iterations):
        for i in range(iterations):
            newbp = self.buildPlan.generate_neighbour()
            new_score = newbp.get_total_score()

            printable = "NEW SCORE {}".format(new_score)

            if (new_score > self.score):
                self.score = new_score
                self.buildPlan = newbp
                printable += " ACCEPTED"
            else:
                printable += " REJECTED"
            print(printable)

        return self.buildPlan

class SimulatedAnnealing(Strategy):

    def algorithm(self, iterations):
        for i in range(iterations):
            
            fraction = i / float(iterations)
            temperature = max(0.01, min(1, 1 - fraction))
            
            newbp = self.buildPlan.generate_neighbour()
            new_score = newbp.get_total_score()
            score_diff = (new_score - self.score)
            temperature = temperature * math.pow(10, math.floor(math.log10(max(abs(score_diff), 1))))
            percentage = math.exp(-abs(score_diff) / temperature)
            random_percentage = random.uniform(0,0.5)
            printable = "NEW SCORE {}".format(new_score)

            if score_diff > 0 or percentage > random_percentage:
                self.buildPlan = newbp
                self.score = new_score
                printable += " ACCEPTED"
                if score_diff < 0:
                    printable += " FROM PROBABILITY"
            else:
                printable += " REJECTED"
            print(printable)

        return self.buildPlan

class TabuSearch(Strategy):
    def __init__(self, buildPlan, min_tabu_tenure, max_tabu_tenure):
        super().__init__(buildPlan)
        self.min_tabu_tenure = min_tabu_tenure
        self.max_tabu_tenure = max_tabu_tenure

    def isTabu(self, score, tabuList):
        router_range = int(self.buildPlan.configs["router-range"])
        value_range = math.pow(10, math.floor(math.log10(max(abs(router_range), 1))))
        for tabu in tabuList:
            if (score >= tabu - value_range and score <= tabu):
                return True
        return False

    def algorithm(self, iterations):
        tabu = [self.buildPlan.total_score]
        tabu_size = self.min_tabu_tenure

        for i in range(iterations):
            while len(tabu) >= tabu_size:
                tabu.pop(0)
            
            newbp = self.buildPlan.generate_neighbour()
            new_score = newbp.get_total_score()

            printable = "NEW SCORE {} ".format(new_score)

            if self.isTabu(new_score, tabu):
                tabu_size = min(tabu_size+1, self.max_tabu_tenure)
                printable += "REJECTED "
                printable += "TABU LIST {} ".format(tabu)
                print(printable)
                continue
            elif new_score > self.score:
                self.buildPlan = newbp
                self.score = new_score
                printable += "ACCEPTED "
            
            tabu_size = max(tabu_size-1, self.min_tabu_tenure)
            tabu.append(new_score)
            printable += "TABU LIST {} ".format(tabu)
            print(printable)

        return self.buildPlan

class GeneticAlgorithm(Strategy):
    def __init__(self, buildPlan, population):
        self.generation = []
        for i in range(population):
            member = buildPlan.copy()
            member.generate_routers()
            member.get_total_score()
            self.generation.append(member)

    def algorithm(self, iterations):

        heapq.heapify(self.generation)
        length = len(self.generation)

        for i in range(iterations):
            parent1 = heapq.nlargest(1, self.generation)[0] # best
            parent2 = heapq.nsmallest(length - 1, self.generation)[random.randint(0, length-2)] # random
            child = self.reproduce(parent1, parent2)
            printable = "NEW CHILD SCORE {} ".format(str(child.total_score))
            printable += "REMOVED SCORE {} ".format(heapq.heapreplace(self.generation, child).total_score)
            printable += "GENERATION {}".format([mb.total_score for mb in self.generation])
            print(printable)
            
        best = heapq.nlargest(1,self.generation)[0]
        return best

    def reproduce(self, parent1, parent2):
        n = random.randint(1, len(parent1.routers) - 1)
        child = set()
        while len(child) < n:
            router_id = random.randint(0, len(parent1.routers)-1)
            child.add(parent1.routers[router_id])

        while len(child) < len(parent1.routers):
            router_id = random.randint(0, len(parent2.routers)-1)
            router = parent2.routers[router_id]
            if random.uniform(0, 1.0) > 0.80:
                router = Router(router.generate_neighbour(parent2), router.rrange)
            child.add(router)
            
        new_bplan = parent1.copy()
        new_bplan.routers = list(child)

        for r in new_bplan.routers:
            new_bplan.map[r.coords] = r

        new_bplan.get_total_score()
        return new_bplan
import random
import math
import heapq

class Strategy():
    def __init__(self, buildPlan):
        self.buildPlan = buildPlan
        self.score = buildPlan.get_total_score()
        print("INITIAL SCORE: ", self.score)

    def algorithm(self, iterations):
        pass

class HillClimbing(Strategy):

    def algorithm(self, iterations):
        new_score = self.score - 1
        failed_tries = 0
        while failed_tries < iterations:
            newbp = self.buildPlan.generate_neighbour()
            new_score = newbp.get_total_score()
            print("NEW SCORE: ", new_score)
            if (new_score > self.score):
                self.score = new_score
                self.buildPlan = newbp
            else:
                failed_tries += 1

        print("FINAL SCORE: ", self.score)
        return self.buildPlan

class SimulatedAnnealing(Strategy):

    def algorithm(self, iterations):
        for i in range(iterations):
            
            fraction = i / float(iterations)
            temperature = max(0.01, min(1, 1 - fraction))
            
            newbp = self.buildPlan.generate_neighbour()
            new_score = newbp.get_total_score()
            score_diff = (new_score - self.score)
            temperature = temperature * math.pow(10,math.floor(math.log10(max(abs(score_diff),1))))
            percentage = math.exp(-abs(score_diff) / temperature)
            random_percentage = random.uniform(0,0.5)
            printable = "PERC {} COMP SCORE {} RD {} ".format(percentage, new_score, random_percentage)

            if score_diff > 0 or percentage > random_percentage:
                self.buildPlan = newbp
                self.score = new_score
                printable += " ACCEPTED"
                if score_diff <= 0:
                    printable += " FROM MATH EXP"
            
            printable += " SCORE {}".format(self.score)
            print(printable)

        return self.buildPlan

class TabuSearch(Strategy):
    def isTabu(self, score, tabuList):
        router_range = int(self.buildPlan.configs["router-range"])
        for tabu in tabuList:
            if score <= tabu + router_range:
                return True
        return False

    def algorithm(self, iterations):
        max_tabu = 5
        tabu = []

        for i in range(iterations):
            
            while len(tabu) >= max_tabu:
                tabu.pop(0)
            
            newbp = self.buildPlan.generate_neighbour()
            new_score = newbp.get_total_score()

            printable = "SCORE {} ".format(new_score)

            if self.isTabu(new_score, tabu):
                max_tabu += 1
                printable += "NOT ACCEPTED "
                print(printable)
                continue
            elif new_score > self.score:
                self.buildPlan = newbp
                self.score = new_score
                printable += "ACCEPTED "
            
            max_tabu = max(max_tabu-1,5)
            tabu.append(new_score)
            printable += "TABU {} ".format(tabu)
        
            print(printable)


        print("FINAL SCORE: ", self.score)
        return self.buildPlan

class GeneticAlgorithm():
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
            parent2 = heapq.nsmallest(length - 1 ,self.generation)[random.randint(0,length-2)] # random
            child = self.reproduce(parent1, parent2)
            print("REMOVING " , heapq.heapreplace(self.generation, child).total_score)
            print("ANOTHER CHILD ",str(child.total_score))
        
        best = heapq.nlargest(1,self.generation)[0]
        print("FINAL SCORE: ", best.total_score)
        return best

    def reproduce(self, parent1, parent2):
        n = random.randint(1, len(parent1.routers) - 1)
        child = set()
        while len(child) < n:
            router_id = random.randint(0, len(parent1.routers)-1)
            child.add(parent1.routers[router_id])

        while len(child) < len(parent1.routers):
            router_id = random.randint(0, len(parent2.routers)-1)
            child.add(parent2.routers[router_id])
            
        new_bplan = parent1.copy()
        new_bplan.routers = list(child)

        for r in new_bplan.routers:
            new_bplan.map[r.coords] = r

        new_bplan.get_total_score()
        return new_bplan
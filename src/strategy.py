import random
import math

class Strategy():
    def __init__(self, buildPlan):
        self.buildPlan = buildPlan
        self.score = buildPlan.get_total_score()
        print("INITIAL SCORE: ", self.score)

    def algorithm(self):
        pass

class HillClimbing(Strategy):

    def algorithm(self):
        new_score = self.score - 1
        failed_tries = 0
        while failed_tries < 20:
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

    def algorithm(self):
        num_iterations = 10

        for i in range(num_iterations):
            
            fraction = i / float(num_iterations)
            temperature = 100*max(0.01, min(1, 1 - fraction))
            
            newbp = self.buildPlan.generate_neighbour()
            new_score = newbp.get_total_score()
            score_diff = new_score - self.score
            printable = "SCORE {} TEMP {}".format(new_score, temperature)

            if score_diff > 0 or math.exp(-score_diff / temperature) > random.uniform(0,1):
                self.buildPlan = newbp
                self.score = new_score
                printable += " ACCEPTED"
                if score_diff <= 0:
                    printable += " FROM MATH EXP"
            
            print(printable)

        return self.buildPlan

class TabuSearch(Strategy):
    def isTabu(self, score, tabuList):
        router_range = int(self.buildPlan.configs["router-range"])
        for tabu in tabuList:
            if score <= tabu + router_range:
                return True
        return False

    def algorithm(self):
        num_iterations = 20
        max_tabu = 5
        tabu = []

        for i in range(num_iterations):
            
            if len(tabu) == max_tabu:
                tabu.pop(0)
            
            newbp = self.buildPlan.generate_neighbour()
            new_score = newbp.get_total_score()

            printable = "SCORE {} ".format(new_score)

            if self.isTabu(new_score, tabu):
                continue
            elif new_score > self.score:
                self.buildPlan = newbp
                self.score = new_score
                printable += "ACCEPTED "
            
            tabu.append(new_score)
            printable += "TABU {} ".format(tabu)
        
            print(printable)


        print("FINAL SCORE: ", self.score)
        return self.buildPlan
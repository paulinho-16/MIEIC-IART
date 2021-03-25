from strategy import HillClimbing, SimulatedAnnealing, TabuSearch
from bplan import BuildPlan
from cell import *
from executor import Executor
from coords import Coords
from router import Router

if __name__ == "__main__":

    bp = BuildPlan("../in/example.data")
    executor = Executor()

    bp.generate_routers()
    
    print([str(r.coords) for r in bp.routers])

    # r1 = Router(Coords(80,90),100)
    # r2 = Router(Coords(220,200),100)
    # r3 = Router(Coords(50,226),100)
    # r4 = Router(Coords(224,26),100)
    # r5 = Router(Coords(127,265),100)
    # bp.routers = [r1,r2,r3, r4, r5]

    # bp.map[r1.coords] = r1
    # bp.map[r2.coords] = r2
    # bp.map[r3.coords] = r3
    # bp.map[r4.coords] = r4
    # bp.map[r5.coords] = r5

    # print(bp.get_draw())

    # print([str(r.coords) for r in bp.generate_neighbour().routers])

    algo = HillClimbing(bp)
    # algo = SimulatedAnnealing(bp)
    # algo = TabuSearch(bp)

    solutionbp = algo.algorithm()

    print([str(r.coords) for r in solutionbp.routers])
    
    print(solutionbp.get_draw())

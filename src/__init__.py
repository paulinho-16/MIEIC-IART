from strategy import HillClimbing, SimulatedAnnealing, TabuSearch, GeneticAlgorithm
from bplan import BuildPlan
from cell import *
from executor import Executor
from coords import Coords
from router import Router
from bparser import Parser
import inquirer
import time

if __name__ == "__main__":

    print(r"""
 ___             _                        ___  _                                   _   
| _ \ ___  _  _ | |_  ___  _ _  ___      | _ \| | __ _  __  ___  _ __   ___  _ _  | |_ 
|   // _ \| || ||  _|/ -_)| '_|(_-/      |  _/| |/ _` |/ _|/ -_)| '  \ / -_)| ' \ |  _|
|_|_\\___/ \_._| \__|\___||_|  /__/      |_|  |_|\__/_|\__|\___||_|_|_|\___||_||_| \__|
    """)
    
    questions = [
        inquirer.List('problem', message="Choose the problem", choices=['example', 'charleston-road', 'plan1', 'rue-de-londres', 'opera', 'lets-go-higher'],),
        inquirer.Text('budget', message="Budget"),
        inquirer.Text('router-range', message="Router Range"),
        inquirer.Text('router-cost', message="Router Cost"),
        inquirer.Text('backbone-cost', message="Backbone Cost"),
        inquirer.List('algorithm', message="Choose the algorithm", choices=['Hill Climbing', 'Simulated Annealing', 'Tabu Search', 'Genetic Algorithm'],),
        inquirer.Text('iterations', message="Iterations"),
    ]

    answers = inquirer.prompt(questions)
    iterations = int(answers["iterations"])
    problem = answers['problem']

    parser = Parser()
    
    parser.image_to_data(
        "./images/{}.png".format(problem), 
        "./in/{}.data".format(problem),
        int(answers["budget"]),
        int(answers["router-range"]),
        int(answers["router-cost"]),
        int(answers["backbone-cost"])
    )

    bp = BuildPlan("./in/{}.data".format(problem))
    executor = Executor()
    algorithm = None

    if answers['algorithm'] == 'Hill Climbing':
        algorithm = HillClimbing(bp)

    elif answers['algorithm'] == 'Simulated Annealing':
        algorithm = SimulatedAnnealing(bp)

    elif answers['algorithm'] == 'Tabu Search':
        tabu_limits = [
            inquirer.Text('min_tabu_tenure', message="Min Tabu Tenure"),
            inquirer.Text('max_tabu_tenure', message="Max Tabu Tenure")
        ]
        answers_tabu = inquirer.prompt(tabu_limits)
        algorithm = TabuSearch(bp, int(answers_tabu['min_tabu_tenure']), int(answers_tabu['max_tabu_tenure']))
        
    elif answers['algorithm'] == 'Genetic Algorithm':
        generation_size = [
            inquirer.Text('generation', message="Generation Dimension")
        ]
        answers_genetic = inquirer.prompt(generation_size)
        algorithm = GeneticAlgorithm(bp, int(answers_genetic['generation']))

    start = time.time()
    if algorithm:
        solutionbp = algorithm.algorithm(iterations)
        print("FINAL SCORE: ", solutionbp.total_score)
        print([str(r.coords) for r in solutionbp.routers])
        output = open("./out/{}.data".format(problem), "w")
        output.write(solutionbp.get_draw())
        output.close()
        parser.data_to_image("./out/{}.data".format(problem),"./out/{}.png".format(problem))
        
    end = time.time()
    print("TOTAL TIME: " + str(end - start))
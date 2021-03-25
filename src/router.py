from math import log10
from coords import Coords
from cell import *
from executor import Executor
from random import randint

class Router(Cell):
    def __init__(self, coords, rrange):
        super().__init__(coords,"R")
        self.rrange = rrange

    def get_covered_targets(self,build_plan):

        def layer_generator(build_plan, layer):
            x = self.coords.x
            y = self.coords.y
            target_cells = []
            wall_cells = []

            for j in range(x - layer, x + layer + 1):
                cell1 = build_plan.get_cell(Coords(j, y+layer))
                cell2 = build_plan.get_cell(Coords(j, y-layer))
                if cell1 and isinstance(cell1, Target):
                    target_cells.append(cell1)
                elif cell1 and isinstance(cell1, Wall):
                    wall_cells.append(cell1)
                if cell2 and isinstance(cell2, Target):
                    target_cells.append(cell2)
                elif cell2 and isinstance(cell2, Wall):
                    wall_cells.append(cell2)

            for j in range(y - layer + 1, y + layer):
                cell1 = build_plan.get_cell(Coords(x+layer, j))
                cell2 = build_plan.get_cell(Coords(x-layer, j))
                if cell1 and isinstance(cell1, Target):
                    target_cells.append(cell1)
                elif cell1 and isinstance(cell1, Wall):
                    wall_cells.append(cell1)
                if cell2 and isinstance(cell2, Target):
                    target_cells.append(cell2)
                elif cell2 and isinstance(cell2, Wall):
                    wall_cells.append(cell2)
            
            return (target_cells, wall_cells)


        def generate_targets(build_plan):
            x = self.coords.x
            y = self.coords.y

            wall_cells = []
            target_cells = []

            with Executor.get_thread_executor() as executor:
                futures = [executor.submit(layer_generator, build_plan, layer) for layer in range(1,self.rrange)]
                result = [f.result() for f in futures]
                for r in result:
                    (t,w) = r
                    target_cells += t
                    wall_cells += w

            final_cells = set()

            for cell in target_cells:
                cx = cell.coords.x
                cy = cell.coords.y

                found = False

                for wall in wall_cells:
                    wx = wall.coords.x
                    wy = wall.coords.y
                    if min(x, cx) <= wx and max(x, cx) >= wx and min(y, cy) <= wy and max(y, cy) >= wy:
                        found = True
                        break
                
                if not found:
                    final_cells.add(cell)
            
            return final_cells
        
        return generate_targets(build_plan)

    def generate_neighbour(self, build_plan):  # Basic "Hill Climbing" (random)
        neighbourhood_range = int(log10(self.rrange) * 5)
        x = self.coords.x
        y = self.coords.y

        nx = x
        ny = y

        obj = build_plan.map.get(Coords(nx,ny), None)

        while (nx,ny) == (x, y) or (obj and type(obj) != Target):
            nx = randint(self.coords.x - neighbourhood_range, self.coords.x + neighbourhood_range)
            ny = randint(self.coords.y - neighbourhood_range, self.coords.y + neighbourhood_range)
            obj = build_plan.map.get(Coords(nx,ny), None)
        
        return Coords(nx,ny)
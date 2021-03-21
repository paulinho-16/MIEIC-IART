from cell import *
from router import Router
from spanning_tree import kruskal
from executor import Executor
import json
import math

class BuildPlan:

    def __init__(self, input_file):
        self.map = dict()
        self.routers = []

        f = open(input_file, 'r')
        lines = f.readlines()

        self.configs = json.loads(lines[0])

        x = 0
        y = 0

        for l in lines[1:]:
            x = 0
            for c in l:
                coords = Coords(x,y)
                if c == "-":
                    self.map[coords] = Void(coords)
                elif c == ".":
                    self.map[coords] = Target(coords)
                elif c == "#":
                    self.map[coords] = Wall(coords)
                elif c == "b":
                    self.map[coords] = Backbone(coords)
                elif c == "R":
                    self.map[coords] = Router(coords)
                elif c == "r":
                    self.map[coords] = Range(coords)
                x += 1
            y += 1
        
        f.close()
        
        self.max_x = x
        self.max_y = y

        self.covered_targets = set()

    def __str__(self, configs = False):
        
        matrix = ""

        if configs:
            matrix += str(self.configs) + "\n"

        for j in range(self.max_y):
            for i in range(self.max_x):
                matrix += str(self.map.get(Coords(i,j),"-"))
            matrix += "\n"
        
        return matrix

    def get_cell(self, coords):
        return self.map.get(coords,False)

    def add_cell(self, cell):
        self.map[cell.coords] = cell

    def get_covered_targets(self):
        covered_targets = set()

        with Executor.get_process_executor() as executor:
            futures = [executor.submit(router.get_covered_targets,self) for router in self.routers]
            results = [f.result() for f in futures]
            for r in results:
                covered_targets.update(r)
        
        self.covered_targets = covered_targets
        
        return covered_targets

    def get_spanning_tree(self):
        def euclid_distance(coords1, coords2):
            x1, y1 = coords1.x, coords1.y
            x2, y2 = coords2.x, coords2.y
            dx, dy = abs(x2-x1), abs(y2-y1)
            _min, _max = min(dx,dy), max(dx,dy)

            diagonal = _min
            straight = _max - _min

            return diagonal + straight
        
        edges = []

        vertices = self.routers + [Backbone(Coords(self.configs["x"],self.configs["y"]))]

        for i in range(len(vertices) - 1):
            for j in range(i+1,len(vertices)):
                c1 = vertices[i].coords
                c2 = vertices[j].coords
                distance = euclid_distance(c1,c2)
                edges.append((i,j,distance))
    
        return kruskal(len(vertices), edges)
        
    def get_backbone_cost(self):
        return self.get_spanning_tree()[0]

    def get_total_cost(self):

        c = len(self.get_covered_targets())
        b = self.configs["budget"]
        nb = self.get_backbone_cost()
        pb = self.configs["backbone-cost"]
        nr = len(self.routers)
        pr = self.configs["router-cost"]

        return 1000 * c + (b - (nb * pb + nr * pr))

    def get_draw(self):
        
        if len(self.covered_targets) == 0:
            self.get_covered_targets()

        def diagonal_line(coords1,coords2):
            
            cells = []
            
            x1, y1 = coords1.x, coords1.y
            x2, y2 = coords2.x, coords2.y
            dx, dy = abs(x2-x1), abs(y2-y1)
            _min, _max = min(dx,dy), max(dx,dy)

            left, right = min(x1,x2), max(x1,x2)
            up, down = max(y1,y2), min(y1,y2)

            if _max == dx:
                ny = y1
                if x2 == left:
                    ny = y2

                while dx > dy:
                    left += 1
                    cells.append(Cell(Coords(left, ny),"b"))
                    dx -= 1

                step_y = 1
                if ny == up:
                    step_y = -1

                while dx > 1:
                    left, ny = left + 1, ny + step_y
                    cells.append(Cell(Coords(left,ny),"b"))
                    dx -= 1
                

            elif _max == dy:
                nx = x1
                if y2 == down:
                    nx = x2

                while dy > dx:
                    down += 1
                    cells.append(Cell(Coords(nx, down),"b"))
                    dy -= 1

                step_x = 1
                if nx == right:
                    step_x = -1

                while dx > 1:
                    nx, down = nx + step_x, down + 1
                    cells.append(Cell(Coords(nx,down),"b"))
                    dx -= 1
            
            return cells                

        targets = list(self.covered_targets)
        
        for c in targets:
            self.map[c.coords] = Cell(c.coords,"r")
        
        vertices = self.routers + [Backbone(Coords(self.configs["x"],self.configs["y"]))]

        tree = self.get_spanning_tree()[1]

        for (v1,v2,_c) in tree:
            _v1 = vertices[v1]
            _v2 = vertices[v2]
            cells = diagonal_line(_v1.coords, _v2.coords)
            for c in cells:
                self.map[c.coords] = c

        return str(self)
from cell import *
from spanning_tree import kruskal
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
        for router in self.routers:
            covered_targets.update(router.get_covered_targets(self))
        return covered_targets
        
    def backbone_cost(self):
        def euclid_distance(coords1, coords2):
            x1 = coords1.x
            y1 = coords1.y

            x2 = coords2.x
            y2 = coords2.y

            dx = abs(x2-x1)
            dy = abs(y2-y1)

            _min = min(dx,dy)
            _max = max(dx,dy)

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
    
        cost = kruskal(len(vertices), edges)

        return cost

    def total_cost(self):

        c = self.get_covered_targets()
        b = self.configs["budget"]
        nb = self.backbone_cost()
        pb = self.configs["backbone-cost"]
        nr = len(self.routers)
        pr = self.configs["router-cost"]

        return 1000 * c + (b - (nb * pb + nr * pr))
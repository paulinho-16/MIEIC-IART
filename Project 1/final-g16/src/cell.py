from coords import Coords

class Cell:

    def __init__(self, coords, symbol):
        self.coords = coords
        self.symbol = symbol

    def __str__(self):
        return self.symbol

    def __eq__(self, value):
        return self.coords.__eq__(value.coords)
    
    def __hash__(self):
        return self.coords.__hash__()

    def get_neighbours(self, build_plan):
        neighbours = []
        x = self.coords.x
        y = self.coords.y
        possible_neighbours = [
            Coords(x, y+1),
            Coords(x, y-1),
            Coords(x+1, y),
            Coords(x-1, y),
            Coords(x+1, y+1),
            Coords(x+1, y-1),
            Coords(x-1, y+1),
            Coords(x-1, y-1)
        ]

        neighbours = []
        for n in possible_neighbours:
            neighbour = build_plan.get_cell(n)
            if neighbour:
                neighbours.append(neighbour)

        return neighbours

class Backbone(Cell):
    def __init__(self, coords):
        super().__init__(coords,"b")

class Wall(Cell):
    def __init__(self, coords):
        super().__init__(coords,"#")

class Range(Cell):
    def __init__(self, coords):
        super().__init__(coords,"r")
        
class Target(Cell):
    def __init__(self, coords):
        super().__init__(coords,".")

class Void(Cell):
    def __init__(self, coords):
        super().__init__(coords,"-")
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

    # dict of type { Coords : Cell , Coords : Cell ... }
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

class Router(Cell):
    def __init__(self, coords, rrange):
        super().__init__(coords,"R")
        self.rrange = rrange

    def get_covered_targets(self,build_plan):
        x = self.coords.x
        y = self.coords.y

        total_cells = set()

        for layer in range(1, self.rrange):
            covered_cells = []

            for j in range(x - layer, x + layer + 1):
                cell1 = build_plan.get_cell(Coords(j, y+layer))
                cell2 = build_plan.get_cell(Coords(j, y-layer))
                if cell1 and not (isinstance(cell1, Router) or isinstance(cell1, Void)):
                    covered_cells.append(cell1)
                if cell2 and not (isinstance(cell2, Router) or isinstance(cell2, Void)):
                    covered_cells.append(cell2)

            for j in range(y - layer + 1, y + layer):
                cell1 = build_plan.get_cell(Coords(x+layer, j))
                cell2 = build_plan.get_cell(Coords(x-layer, j))
                if cell1 and not (isinstance(cell1, Router) or isinstance(cell1, Void)):
                    covered_cells.append(cell1)
                if cell2 and not (isinstance(cell2, Router) or isinstance(cell2, Void)):
                    covered_cells.append(cell2)

            print("Camada:",str(layer),len(covered_cells))
            print([str(c.coords) for c in covered_cells])
            total_cells.update(covered_cells)
        
        total = len(total_cells)
        count = 0

        a = self.coords.x
        b = self.coords.y

        target_cells = set()

        lt_cells = list(total_cells)

        while count < total:
            c = lt_cells[count]
            if isinstance(c,Wall):
                w = c.coords.x
                v = c.coords.y
                i = count + 1
                while i < total:
                    nc = lt_cells[i]
                    x = nc.coords.x
                    y = nc.coords.y
                    if isinstance(nc,Target) and min(a, x) <= w and max(a, x) >= w and min(b, y) <= v and max(b, y) >= v:
                        target_cells.add(nc)
                    i += 1
            count += 1

        return list(total_cells.difference(target_cells))
                
    # --------------------------------------------------------------------------------------

    def get_covered_targets2(self, build_plan):
        covered_cells = []
        x = self.coords.x
        y = self.coords.y
        for i in range(x - self.rrange, x + self.rrange + 1):
            for j in range(y - self.rrange, y + self.rrange + 1):
                cell = build_plan.get_cell(Coords(i, j))
                if cell:
                    covered_cells.append(cell)
    
        covered_targets = []
        for cell in covered_cells:
            if cell.symbol == "." and not self.wall_block(cell, covered_cells):
                covered_targets.append(cell)
        return covered_targets
    
    def wall_block(self, cell, covered_cells):
        for adj in covered_cells:
            if not isinstance(adj, Wall):
                continue
            a = self.coords.x
            b = self.coords.y

            x = cell.coords.x
            y = cell.coords.y

            w = adj.coords.x
            v = adj.coords.y
            
            if min(a, x) <= w and max(a, x) >= w and min(b, y) <= v and max(b, y) >= v:
                return True
        return False

class Range(Cell):
    def __init__(self, coords):
        super().__init__(coords,"r")
        
class Target(Cell):
    def __init__(self, coords):
        super().__init__(coords,".")

class Void(Cell):
    def __init__(self, coords):
        super().__init__(coords,"-")
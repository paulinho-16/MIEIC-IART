from bplan import BuildPlan
from cell import *
from coords import Coords

if __name__ == "__main__":
    bp = BuildPlan("../in/example.data")

    r = Router(Coords(15,2),4)
    bp.routers = [r]

    bp.map[r.coords] = r

    for c in bp.get_covered_targets():
        if not isinstance(c, Router) and not isinstance(c, Wall):
            bp.map[c.coords] = Cell(c.coords,"X")
    
    print(bp)
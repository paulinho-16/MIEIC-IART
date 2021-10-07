from PIL import Image
from numpy import array, asarray, array_equal, savetxt
from json import dumps

class Parser():

    def __init__(self):
        self.GREY = array([170,170,170,255])
        self.WHITE = array([255,255,255,255])
        self.BLACK = array([0,0,0,255])
        self.GREEN = array([37,255,0,255])
        self.BLUE = array([0,0,255,255])
        self.LBLUE = array([225,225,255,255])

    def image_to_data(self, input_file, output_file, budget, router_range, router_cost, backbone_cost):
        config = {"budget": budget, "router-range": router_range, "router-cost": router_cost, "backbone-cost": backbone_cost}

        # loads the image
        image = Image.open(input_file)

        # converts image to numpy array
        data = asarray(image)

        # backbone coordinates
        bcoords = {"x": 0, "y": 0}
        x = 0
        y = 0

        matrix = ""

        # generates the ASCII codes matching each pixel color
        for i in data:
            line = ""
            for j in i:
                if array_equal(j,self.GREY):
                    line += "-"
                elif array_equal(j,self.WHITE):
                    line += "."
                elif array_equal(j,self.BLACK):
                    line += "#"
                elif array_equal(j,self.GREEN):
                    line += "b"
                    bcoords["x"] = x
                    bcoords["y"] = y
                elif array_equal(j,self.BLUE):
                    line += "R"
                elif array_equal(j,self.LBLUE):
                    line += "r"
                x += 1
            y += 1
            x = 0
            matrix += line + "\n"

        config["x"] = bcoords["x"]
        config["y"] = bcoords["y"]

        # saves the configurations in the first line
        matrix = dumps(config) + "\n" + matrix

        # opens the file descriptor
        f = open(output_file,"w")

        # saves the matrix info
        f.write(matrix)

        # closes the file descriptor
        f.close()

    def data_to_image(self, input_file, output_file):
        pixels = []

        # opens the file descriptor
        f = open(input_file, 'r')

        # gets an array of lines
        lines = f.readlines()

        # generates the pixels colors matching each character
        for l in lines:
            nl = []
            for c in l:
                if c == "-":
                    nl.append(self.GREY)
                elif c == ".":
                    nl.append(self.WHITE)
                elif c == "#":
                    nl.append(self.BLACK)
                elif c == "b":
                    nl.append(self.GREEN)
                elif c == "R":
                    nl.append(self.BLUE)
                elif c == "r":
                    nl.append(self.LBLUE)
            pixels.append(nl)
        
        # converts to a numpy array
        pixels = array(pixels)

        # converts the array of pixels to an image
        image = Image.fromarray(pixels.astype("uint8"), 'RGBA')

        # saves the image
        image.save(output_file)
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Building Plan Image Parser
# 
# The following code allows to parse an image of a Building Plan to an ASCII representation that can be used as input for the algorithm to be developed in the Router Placement Optimization Problem.
# 
# An example image is the following:
# 
# ![Building Plan](../images/plan1.png)
# 
# Every pixel color will be translated to an ASCII code by the following table:
# 
# | Color                 | Symbol | Meaning |
# |:------------------------|:---:|---------:|
# | GREY (170,170,170,255)  | `-` | void     |
# | WHITE (255,255,255,255) | `.` | valid    |
# | BLACK (0,0,0,255)       | `#` | wall     |
# | GREEN (37,255,0,255)    | `b` | backbone |
# | BLUE (0,0,255,255)      | `R` | router   |
# | LBLUE (225,225,255,255) | `r` | coverage |
# 
# After running the algorithm and getting a solution represented by an ASCII matrix, it can be translated back to an image, like the following:
# 
# ![Building Plan Router Placement](../images/plan1r.png)
# %% [markdown]
# Getting the environment ready by doing:
# 
# ```bash
# python3 -m venv env
# source env/bin/activate
# pip install -r requirements.txt
# ```
# 
# Start the jupiter-lab and select the Python kernel at `env/bin/python`
# 
# With that done, first, let's start by importing the necessary modules:

# %%
from PIL import Image
from numpy import array, asarray, array_equal, savetxt

# %% [markdown]
# Then, create the color constants to be mapped to the characters:

# %%
GREY = array([170,170,170,255])
WHITE = array([255,255,255,255])
BLACK = array([0,0,0,255])
GREEN = array([37,255,0,255])
BLUE = array([0,0,255,255])
LBLUE = array([225,225,255,255])

# %% [markdown]
# After that, create a function to convert an image to its ASCII representation, along with the problem configurations:
# 
# 1. Start by loading the image with the Image object from the Pillow module.
# 2. Convert the image object to a numpy array.
# 3. Save the configurations as the first line of the data.
# 4. Iterate over each pixel and map its color to a character from the table.
# 5. Save the new data to the ouput file.

# %%
def image_to_data(input_file, output_file, budget, router_range, router_cost, backbone_cost):
    config = {"budget": budget, "router-range": router_range, "router-cost": router_cost, "backbone-cost": backbone_cost}

    # loads the image
    image = Image.open(input_file)

    # converts image to numpy array
    data = asarray(image)

    # saves the configurations in the first line
    matrix = str(config) + "\n"

    # generates the ASCII codes matching each pixel color
    for i in data:
        line = ""
        for j in i:
            if array_equal(j,GREY):
                line += "-"
            elif array_equal(j,WHITE):
                line += "."
            elif array_equal(j,BLACK):
                line += "#"
            elif array_equal(j,GREEN):
                line += "b"
            elif array_equal(j,BLUE):
                line += "R"
            elif array_equal(j,LBLUE):
                line += "r"
        matrix += line + "\n"

    # opens the file descriptor
    f = open(output_file,"w")

    # saves the matrix info
    f.write(matrix)

    # closes the file descriptor
    f.close()

# %% [markdown]
# Next, create a function to convert the ASCII solution representation back to an image:
# 
# 1. Start by loading the data, reading it line by line.
# 2. Convert each character to the corresponding color.
# 3. Convert the entire array to a numpy array.
# 4. Convert the numpy array to an Image object from the Pillow module.
# 5. Save the image to the output file.

# %%
def data_to_image(input_file, output_file):
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
                nl.append(GREY)
            elif c == ".":
                nl.append(WHITE)
            elif c == "#":
                nl.append(BLACK)
            elif c == "b":
                nl.append(GREEN)
            elif c == "R":
                nl.append(BLUE)
            elif c == "r":
                nl.append(LBLUE)
        pixels.append(nl)
    
    # converts to a numpy array
    pixels = array(pixels)

    # converts the array of pixels to an image
    image = Image.fromarray(pixels.astype("uint8"), 'RGBA')

    # saves the image
    image.save(output_file)

# %% [markdown]
# As an example, convert the image 
# 
# ![](../images/plan1.png) 
# 
# to its ASCII representation, saving it to `out/plan1.data`. The problem configurations can be:
# 
# * Budget = 5000
# * Router Range = 100
# * Cost Per Router = 200
# * Cost Per Backbone = 1

# %%
image_to_data("../images/plan1.png","../out/plan1.data",5000,100,200,1)

# %% [markdown]
# As another example, convert the ASCII data in the file `out/plan1.data` back to its image representation, saving it to `out/plan1.png`:
# 
# > Before executing this line, remove the first line of this file, containing the previously assigned configurations.

# %%
data_to_image("../out/plan1r.data","../out/plan1r.png")



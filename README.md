# Router Placement Optimization

* Eduardo Brito up201806271@fe.up.pt
* Paulo Ribeiro up201806505@fe.up.pt
* Rita Silva up201806527@fe.up.pt

### The Problem:

Given a building plan, a decision needs to be made on:

> Where to put wireless routers and how to connect them to the ﬁber backbone to maximize coverage and minimize cost.

### Formulation as an Optimization Problem:

#### Solution Representation

Given a building blueprint, represented as a character matrix, the solution will consist of the same matrix of the building, with the routers and the backbone on their respective positions.

| ![](images/plan1.png) | ![](images/plan1r.png) |
| --------------------- | ---------------------- |

The cells used to link the backbone and the router will be highlighted, along with the cells inside the router range.

#### Rigid Constraints

* To start, there is only one cell connected to the backbone, which can be of any type - target(`.`), void(`-`) or wall(`#`).
* A router can’t be placed inside a wall and it has to be connected to the ﬁber backbone - a cable that delivers Internet to the router itself.
* Each router covers a square area of at most $$(2×R+1)^2$$ cells around it, unless the signal is stopped by a wall cell:
  * $| a − r_x | ≤ R$ , and $| b − r_y | ≤ R$
  * there is no wall cell inside the smallest enclosing rectangle of $[a, b]$ and $[r_x, r_y]$. 
  That is, there is no wall cells $[w, v]$ where both $min(a, r_x) ≤ w ≤ max(a, r_x)$ and $min(b, r_y) ≤ v ≤ max(b, r_y)$.
* Placing a single router costs $P_r$ and connecting a single cell to the backbone costs $P_b$. To ensure the budget constraint, the cost to place the routers and its connections cannot be higher than the budget: $N_b × P_b + N_r × P_r ≤ B$

#### Neighborhood/Mutation

One possible new-solution generator may be to start by randomly placing a number N of routers. 

The solution neighborhood is represented by the neighboring cells of each one of the routers’ positions.

#### Crossover Functions

The crossover between two distinct solutions will consist on selecting and combining the routers with the highest coverage from each solution, without overlapping ranges.

#### Evaluation Function

Being c the total number of target cells covered, the
value of a solution can be computed as follows: 

$value = c + ( B − ( N_b × P_b + N_r × P_r ))$ 

where:
* $B$ is the Budget
* $N_b$ is the number of Backbone cells used
* $P_b$ is the cost per Backbone cell
* $N_r$ is the number of Routers used
* $P_r$ is the cost per Router

A solution found to have an higher value for a given building plan may be a solution that covers more target cells, while respecting the budget and having lower implementation costs.

#### Resolution Workﬂow

||              ||
|----|:---------|----|
| ![](images/plan1.png) | 1. ASCII Conversion<br>2. Algorithm Application<br>3. Solution Generation<br>4. Image Conversion | ![](images/plan1r.png) |

| Color                 | Symbol | Meaning |
|:------------------------|:---:|---------:|
| GREY (170,170,170,255)  | `-` | void     |
| WHITE (255,255,255,255) | `.` | valid    |
| BLACK (0,0,0,255)       | `#` | wall     |
| GREEN (37,255,0,255)    | `b` | backbone |
| BLUE (0,0,255,255)      | `R` | router   |
| LBLUE (225,225,255,255) | `r` | coverage |

#### Tools

##### Programming Language
Python3

##### Development Environment

JupyterLab & Visual Studio Code

##### Data Structures

The information will be converted to a HashMap, whose keys are the coordinate pairs and the values the different cell types.

##### File Structure

The input ﬁle may be an image or an ASCII representation of the building, with only the `void`, `target` and `wall` cells. It can also contain the information about the budget and the costs. The solution ﬁle will consist on the same schema, but with the router(s) placed, along with the cells within the range and the backbone links. An image of the solution will be generated, with the representative colors.

#### References and Related Work

* Mohammed A. Alanezi , Houssem R. E. H. Bouchekara, and Muhammad S. Javaid. (2020). Optimizing Router Placement of Indoor Wireless Sensor Networks in Smart Buildings for IoT Applications.
* Google #Hash Code 2017 Challenge "Router placement" - Team Gyrating Flibbittygibbitts.
* Introduction to Mutation, Crossover Operators, Survivor Selection - Genetic Algorithms, TutorialsPoint
* Google #Hash Code 2018 Challenge “ City Plan - Optimization Problem for Public Projects Implementation”
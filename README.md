## Assign 5

### Coding directions

In this assignment I implemented a Genetic Algorithm to solve the Traveling Salesperson
problem for a complete, weighted, and directed graph. 

### Required: Function implemented for Assignment

* __TSPwGenAlgo(W, population_size=50, mutation_rate=0.01, explore_rate=0.5)__
  * input: an adjacency weight matrix, maximum number of generations, population
    size to use, mutation rate, and exploration rate (lower exploration rate
    means a smaller group of 'fit' individuals should be used for reproduction,
    a larger exploration rate means a larger group of individuals should be used)
  * output/return: a dictionary containing the solution path and distance, and
    a list with the shortest distance found in each generation (see assign5.py) 

"""
For this assignment there is no automated testing. You will instead submit
your *.py file in Canvas. I will download and test your program from Canvas.
"""
import random
import sys
import time
import heapq
INF = sys.maxsize

def adjMatFromFile(filename):
    """ Create an adj/weight matrix from a file with verts, neighbors, and weights. """
    f = open(filename, "r")
    n_verts = int(f.readline())
    print(f" n_verts = {n_verts}")
    adjmat = [[None] * n_verts for i in range(n_verts)]
    for i in range(n_verts):
        adjmat[i][i] = 0
    for line in f:
        int_list = [int(i) for i in line.split()]
        vert = int_list.pop(0)
        assert len(int_list) % 2 == 0
        n_neighbors = len(int_list) // 2
        neighbors = [int_list[n] for n in range(0, len(int_list), 2)]
        distances = [int_list[d] for d in range(1, len(int_list), 2)]
        for i in range(n_neighbors):
            adjmat[vert][neighbors[i]] = distances[i]
    f.close()
    return adjmat


def TSPwGenAlgo(g, max_num_generations=500, population_size=250,
        mutation_rate=0.04, explore_rate=0.75):
    """ A genetic algorithm to attempt to find an optimal solution to TSP  """

    solution_cycle_distance = None # the distance of the final solution cycle/path
    solution_cycle_path = [] # the sequence of vertices representing final sol path to be returned
    shortest_path_each_generation = [] # store shortest path found in each generation

     
    # create and initialize initial members of the population to a 'solution'
    population = generatePop(len(g), population_size)
    
    gensCompleted = 0
    # loop for x number of generations (with possibly other early-stopping criteria)
    for gen in range(max_num_generations):
        scoredPops = []
        # calculate fitness of each individual in the population
        for pop in population:
            heapq.heappush(scoredPops, (calcPath(g, pop), pop))
        
        # append distance and path of the 'fittest' to shortest_path_each_generation
        shortest_path_each_generation.append(scoredPops[0][0])

        # create pool of parents from pops with valid solutions
        parents = []
        total_score = 0
        for sPop in scoredPops:
            if sPop[0] == INF:
                break
            total_score += 1 / sPop[0]
            parents.append((total_score, sPop[1]))

        # Generate pairs of parents
        numCouples = int(population_size * explore_rate) // 2
        couples = genCouples(numCouples, parents)
        
        newGen = []
        # create individuals of the new generation (using some form of crossover)
        for couple in couples:
            newGen.extend(makeChild(couple))

        # fill in rest of new pop with highest performing from last gen
        while len(newGen) < population_size:
            newGen.append(heapq.heappop(scoredPops)[1])
        
        # allow for mutations
        mutations = 0
        for pop in newGen:
            if random.random() <= mutation_rate:
                swap1 = random.randint(1, len(g)-1)
                swap2 = random.randint(1, len(g)-1)
                while swap2 == swap1: # Ensure mutation actually changes pop
                    swap2 = random.randint(1, len(g)-1)
                pop[swap1], pop[swap2] = pop[swap2], pop[swap1]
                mutations += 1
        population = newGen
        gensCompleted += 1
        # print(f"Generation {gen} completed")
        # print(f"    Best Path: {shortest_path_each_generation[gen]}")
        # print(f"    Mutations: {mutations}")
        
    final_scored_Pops = []
    # calculate fitness of each individual in the final population
    for pop in population:
        heapq.heappush(final_scored_Pops, (calcPath(g, pop), pop))
    heapq.heapify(final_scored_Pops)
    # update solution_cycle_distance and solution_path
    solution_cycle_distance = final_scored_Pops[0][0]
    solution_cycle_path = final_scored_Pops[0][1]

    return {
            'solution': solution_cycle_path,
            'solution_distance': solution_cycle_distance,
            'evolution': shortest_path_each_generation
           }


def generatePop(g_size, pop_size):
    """ A helper function for the genetic algorithm to create an initial pop """
    population = [None]*pop_size # Create array to hold the entire population
    # create individual members of the population
    for i in range(pop_size):
        pop = list(range(1, g_size)) # create array filled with vertices
        random.shuffle(pop) # randomize path
        pop.insert(0, 0) # add starting vertex
        population[i] = pop # add pop to pool
    return population


def calcPath(g, path):
    """ Calculates the length of the path on graph g """
    path_length = 0 # initialize path length at 0
    prev_node = path[0] # initilize pervious node as starting node
    # Calculate path lenght along graph
    for node in path[1:]:
        if g[prev_node][node] is not None: # check if path is valid
            path_length += g[prev_node][node]
        else:
            return INF
        prev_node = node
    if g[path[-1]][path[0]] is not None:
        path_length += g[path[-1]][path[0]] # add distance to get back to start
    else:
        return INF
    return path_length


def genCouples(numCouples, parents):
    """ Produces list of numCouples couples given a list of tuples
                containing a stepped binning probability and a parent """
    couples = [None]*numCouples
    for i in range(numCouples):
        couples[i] = genCouple(parents)
    return couples
                    

def genCouple(parents):
    """ Produces a couple given a list of tuples 
        containing stepped probability and a parent """
    if len(parents) < 2:
        return [parents[0][1], parents[0][1]]
    elif len(parents) == 2:
        return [parents[0][1], parents[1][1]]
    couple = [None]*2
    firstP = random.random()*parents[-1][0]
    selection = len(parents) // 2
    step = selection // 2
    while True: # Select First Parent
        if firstP <= parents[selection][0]:
            if selection == 0 or firstP > parents[(selection-1)][0]:
                couple[0] = parents[selection][1]
                break
            else:
                if selection - step < 0:
                    selection = 0
                else:
                    selection -= step
                step = (step + 1) // 2
        else:
            if selection + step >= len(parents):
                selection = len(parents)-1
            else:
                selection += step
            step = (step + 1) // 2
        if step == 0:
            print("Error: Step=0 when selecting Parent 1, exiting...")
            quit()
    firstP = selection # save this value so we can easily check for duplicate parents
    secondP = random.random()*parents[-1][0]
    selection = len(parents) // 2
    step = selection // 2
    # Select Second Parent
    while True:
        if secondP <= parents[selection][0]:
            if selection == 0 or secondP > parents[(selection-1)][0]:
                if selection == firstP:
                    # Duplicate parent, restart search
                    secondP = random.random()*parents[-1][0]
                    selection = len(parents) // 2
                    step = selection // 2
                else:
                    couple[1] = parents[selection][1]
                    break
            else:
                if selection - step < 0:
                    selection = 0
                else:
                    selection -= step
                step = (step + 1) // 2
        else:
            if selection + step >= len(parents):
                selection = len(parents)-1
            else:
                selection += step
            step = (step + 1) // 2

        if step == 0:
            print("Error: Step=0 when selecting Parent 2, exiting...")
            quit()
    return couple


def makeChild(couple):
    """ Create two child paths from a tuple of two parent paths """
    children = [[], []]
    subsetLen = len(couple[0]) // 2
    offset = random.randint(1, subsetLen-2) # add a bit of randomization
    subsets = [None]*2
    for i in range(2):
        subsets[i] = couple[i][offset:(offset+subsetLen)]
    for i in range(2):
        for j, vert in enumerate(couple[i]):
            if vert not in subsets[i-1]:
                children[i].append(vert)
            if j == offset:
                children[i].extend(subsets[i-1])
    
    return children


def TSPwDynProg(g):
    """ (10pts extra credit) A dynamic programming approach to solve TSP """
    solution_cycle_distance = None # the distance of the final solution cycle/path
    solution_cycle_path = [] # the sequence of vertices representing final sol path to be returned

    #...

    return {
            'solution': solution_cycle_path,
            'solution_distance': solution_cycle_distance,
           }


def TSPwBandB(g):
    """ (10pts extra credit) A branch and bound approach to solve TSP """
    solution_cycle_distance = None # the distance of the final solution cycle/path
    solution_cycle_path = [] # the sequence of vertices representing final sol path to be returned

    #...

    return {
            'solution': solution_cycle_path,
            'solution_distance': solution_cycle_distance,
           }


def big_gen_test(num_tests):
    """ Do multiple runs at once """
    g = adjMatFromFile("complete_graph_n100.txt")
    max_gens = 500
    pop_size = 250
    mt_rate = 0.04
    ex_rate = 0.7
    results = [None]*num_tests
    for i in range(num_tests):
        results[i] = TSPwGenAlgo(g, max_gens, pop_size, mt_rate, ex_rate)['solution_distance']
    print(f"Ran genetic algorithm {num_tests} times @ max_gens: {max_gens} pop_size: {pop_size} mutation_rate: {mt_rate} explore_rate: {ex_rate}")
    print(f"    Best score: {min(results)}")
    print(f"    worst score: {max(results)}")
    print(f"    Avarage score: {sum(results)/num_tests}")


def assign05_main():
    """ Load the graph (change the filename when you're ready to test larger ones) """
    g = adjMatFromFile("complete_graph_n100.txt")

    # Run genetic algorithm to find best solution possible
    start_time = time.time()
    res_ga = TSPwGenAlgo(g)
    elapsed_time_ga = time.time() - start_time
    print(f"GenAlgo runtime: {elapsed_time_ga:.2f}")
    print(f"  sol dist: {res_ga['solution_distance']}")
    print(f"  sol path: {res_ga['solution']}")
    print(f"  sol evolution: {res_ga['evolution']}")

    # (Try to) run Dynamic Programming algorithm only when n_verts <= 10
    if len(g) <= 10:
        start_time = time.time()
        res_dyn_prog = TSPwDynProg(g)
        elapsed_time = time.time() - start_time
        if len(res_dyn_prog['solution']) == len(g) + 1:
            print(f"Dyn Prog runtime: {elapsed_time:.2f}")
            print(f"  sol dist: {res_dyn_prog['solution_distance']}")
            print(f"  sol path: {res_dyn_prog['solution']}")

    # (Try to) run Branch and Bound only when n_verts <= 10
    if len(g) <= 10:
        start_time = time.time()
        res_bnb = TSPwBandB(g)
        elapsed_time = time.time() - start_time
        if len(res_bnb['solution']) == len(g) + 1:
            print(f"Branch & Bound runtime: {elapsed_time:.2f}")
            print(f"  sol dist: {res_bnb['solution_distance']}")
            print(f"  sol path: {res_bnb['solution']}")


# Check if the program is being run directly (i.e. not being imported)
if __name__ == '__main__':
    big_gen_test(20)


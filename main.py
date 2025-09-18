import os
import random
import argparse

import numpy as np

from classes.evolutionary_agent import EvolutionaryAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sudoku solver that uses evolutionary algorithms')
    parser.add_argument('--sudoku-path', type=str, help='The path to the sudoku you would like solved', default='/home/troyxdp/Documents/University Work/Optimization/Project/test_data/1.txt')
    parser.add_argument('--mutation-rate', type=float, help='The mutation rate to use for the evolutionary algorithm - 0.02 is recommended', default=0.03)
    parser.add_argument('--nt', type=int, help='Size of subset to be used for tournament in tournament selection', default=12)
    parser.add_argument('--wait', type=int, help='Number of generations to wait for an improvement before terminating', default=300)
    parser.add_argument('--num-offspring', type=int, help='Number of offspring to generate per generation', default=4500)
    args = parser.parse_args()

    # Get sudoku
    grid = []
    with open(args.sudoku_path, 'r') as f:
        grid_txt = f.readlines()
        grid_txt = [row.strip() for row in grid_txt]
        for row in grid_txt:
            grid_row = row.split(' ')
            grid_row = [int(num) for num in grid_row]
            grid.append(grid_row)

    # Create evolutionary agent
    ea_agent = EvolutionaryAgent(np.array(grid))
    solution = ea_agent.run_evolutionary_algorithm(mutation_rate=args.mutation_rate, no_improvement_max_generations=args.wait, n_t=args.nt, num_offspring=args.num_offspring)
    if solution:
        print("Found solution!\n")
        print(solution)
    else:
        print("No solution found!")
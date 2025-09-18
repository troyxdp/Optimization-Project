import os
import random
import argparse

import numpy as np

from classes.evolutionary_agent import EvolutionaryAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sudoku solver that uses evolutionary algorithms')
    parser.add_argument('--sudoku-path', type=str, help='The path to the sudoku you would like solved', default='/home/troyxdp/Documents/University Work/Optimization/Project/test_data/1.txt')
    parser.add_argument('--mutation-rate', type=float, help='The mutation rate to use for the evolutionary algorithm - 0.02 is recommended', default=0.02)
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
    ea_agent.run_evolutionary_algorithm(mutation_rate=args.mutation_rate)
import argparse
import ast

import numpy as np

from classes.sudoku_evolutionary_agent import StandardSudokuEvolutionaryAgent, KillerSudokuEvolutionaryAgent
from classes.sudoku import Cage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sudoku solver that uses evolutionary algorithms')
    parser.add_argument('--sudoku-path', type=str, help='The path to the sudoku you would like solved', default='/home/troyxdp/Documents/University Work/Optimization/Project/test_data/a.txt')
    parser.add_argument('--mutation-rate', type=float, help='The mutation rate to use for the evolutionary algorithm - 0.02 is recommended', default=0.03)
    parser.add_argument('--nt', type=int, help='Size of subset to be used for tournament in tournament selection', default=12)
    parser.add_argument('--wait', type=int, help='Number of generations to wait for an improvement before terminating', default=50)
    parser.add_argument('--num-offspring', type=int, help='Number of offspring to generate per generation', default=4500)
    parser.add_argument('--output-folder', type=str, help='Path to output CSV stat files to', default='')
    parser.add_argument('--killer', action='store_true', help='Flag for whether to do a killer sudoku or normal sudoku')
    args = parser.parse_args()

    if args.killer:
        # Get sudoku
        grid = []
        with open(args.sudoku_path, 'r') as f:
            grid_txt = f.readlines()
            grid_txt = [row.strip() for row in grid_txt]
            for row in grid_txt:
                grid_row = row.split(' ')
                grid_row = [int(num) for num in grid_row]
                grid.append(grid_row)

        # Create evolutionary agent and solve puzzle using agent
        ea_agent = StandardSudokuEvolutionaryAgent(np.array(grid))
        ea_agent.run_evolutionary_algorithm(mutation_rate=args.mutation_rate, no_improvement_max_generations=args.wait, n_t=args.nt, num_offspring=args.num_offspring, output_folder=args.output_folder)
        solution = ea_agent.get_solution()
        if solution:
            print("Found solution!\n")
            print(solution)
        else:
            print("No solution found!")
    else:
        cages = []
        with open(args.sudoku_path, 'r') as f:
            cells_txt = f.readlines()
            cells_txt = [row.strip() for row in cells_txt]
            for row in cells_txt:
                cells_row = row.split(' ')
                correct_sum = int(cells_row[0])
                cells = [ast.literal_eval(cell_txt) for cell_txt in cells_row[1:]]
                cage = Cage(cells, correct_sum)
                cages.append(cage)
        
        # Create evolutionary agent
        ea_agent = KillerSudokuEvolutionaryAgent(np.zeros((9, 9)), cages)
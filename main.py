import argparse
import ast

import numpy as np

from classes.sudoku_evolutionary_agent import StandardSudokuEvolutionaryAgent, KillerSudokuEvolutionaryAgent, SudokuXEvolutionaryAgent
from classes.sudoku import Cage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sudoku solver that uses evolutionary algorithms')
    parser.add_argument('--sudoku-path', type=str, help='The path to the sudoku you would like solved', default='/home/troyxdp/Documents/University Work/Optimization/Project/test_data/1.txt')
    parser.add_argument('--mutation-rate', type=float, help='The mutation rate to use for the evolutionary algorithm - 0.03 is recommended', default=0.03)
    parser.add_argument('--nt', type=int, help='Size of subset to be used for tournament in tournament selection', default=12)
    parser.add_argument('--no-improvement-max-generations', type=int, help='Number of generations to wait for an improvement before terminating', default=25)
    parser.add_argument('--num-mutation-retries', type=int, help='Number of times to inject randomness upon premature convergence', default=1)
    parser.add_argument('--mutation-retry-scale', type=float, help='Value to scale mutation rate by when injecting randomness', default=2)
    parser.add_argument('--num-offspring', type=int, help='Number of offspring to generate per generation', default=4500)
    parser.add_argument('--output-folder', type=str, help='Path to output CSV stat files to', default='')
    parser.add_argument('--killer', action='store_true', help='Flag for whether to do a killer sudoku or normal sudoku')
    parser.add_argument('--sudoku-x', action='store_true', help='Flag for whether to do normal sudoku or sudoku X')
    parser.add_argument('--population-size', type=int, help='Size of population in each generation', default=8100)
    args = parser.parse_args()

    if not args.killer and not args.sudoku_x:
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
        ea_agent.run_evolutionary_algorithm(
            mutation_rate=args.mutation_rate, 
            num_offspring=args.num_offspring, 
            n_t=args.nt, 
            no_improvement_max_generations=args.no_improvement_max_generations, 
            max_mutation_retries=args.num_mutation_retries,
            mutation_retry_scale=args.mutation_retry_scale,
            output_folder=args.output_folder
        )
        solution = ea_agent.get_solution()
        if not solution is None:
            print("Found solution!\n")
            print(solution)
        else:
            print("No solution found!")

    elif args.sudoku_x:
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
        ea_agent = SudokuXEvolutionaryAgent(np.array(grid))
        ea_agent.run_evolutionary_algorithm(
            mutation_rate=args.mutation_rate, 
            num_offspring=args.num_offspring, 
            n_t=args.nt, 
            no_improvement_max_generations=args.no_improvement_max_generations, 
            max_mutation_retries=args.num_mutation_retries,
            mutation_retry_scale=args.mutation_retry_scale,
            output_folder=args.output_folder
        )
        solution = ea_agent.get_solution()
        if not solution is None:
            print("Found solution!\n")
            print(solution)
        else:
            print("No solution found!")

    elif args.killer:
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
        
        # Create evolutionary agent and solve puzzle using agent
        ea_agent = KillerSudokuEvolutionaryAgent(np.zeros((9, 9), dtype=np.int8), cages)
        ea_agent.run_evolutionary_algorithm(
            mutation_rate=args.mutation_rate, 
            num_offspring=args.num_offspring, 
            n_t=args.nt, 
            no_improvement_max_generations=args.no_improvement_max_generations, 
            max_mutation_retries=args.num_mutation_retries,
            mutation_retry_scale=args.mutation_retry_scale,
            output_folder=args.output_folder
        )
        solution = ea_agent.get_solution()
        if not solution is None:
            print("Found solution!\n")
            print(solution)
        else:
            print("No solution found!")
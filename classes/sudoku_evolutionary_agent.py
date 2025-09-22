import os
import random
from copy import deepcopy
import math
from abc import ABC, abstractmethod

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import yaml

from classes.sudoku import *

class SudokuEvolutionaryAgent(ABC):
    # CONSTRUCTOR
    def __init__(self, population_size: int):
        # Value to initialize inside this parent class
        self._population_size = population_size

        # Common values initialized inside each sub class' constructor
        self._solution = None
        self.max_possible_fitness = -math.inf
        self.population = []

    # GETTER METHODS
    def get_solution(self):
        return self._solution

    # FUNCTIONAL METHODS
    def run_evolutionary_algorithm(
            self, 
            mutation_rate:float=0.02, 
            num_offspring:int=4500, 
            n_t:int=12,  
            no_improvement_max_generations:int=50,
            max_mutation_retries:int=2,
            mutation_retry_scale:float=1.5,
            output_folder:str=''
        ):
        if output_folder.strip():
            # Write hyperparameters to YAML
            hyperparams = {
                'mutation_rate': mutation_rate,
                'num_offspring': num_offspring,
                'tournament_size': n_t,
                'wait': no_improvement_max_generations,
                'max_mutation_retries': max_mutation_retries,
                'mutation_retry_scale': mutation_retry_scale
            }
            with open(os.path.join(output_folder, 'hyperparams.yaml'), 'w') as f:
                yaml.dump(hyperparams, f)

        # Initialize values
        max_fitness = 0
        last_improvement_generation = 0
        mutation_retries = 0
        curr_generation_num = 0
        total_fitnesses_per_generation = []
        max_fitness_over_generations = []
        best_fitness_per_generation = []
        max_total_fitness = 0

        # Start evolutionary process
        print("Starting evolution...")
        print(f"Maximum possible fitness: {self.max_possible_fitness}\n")
        while not max_fitness == self.max_possible_fitness and mutation_retries <= max_mutation_retries:
            # If no improvement over a certain number of generations, inject some randomness into the population
            if curr_generation_num - last_improvement_generation >= no_improvement_max_generations:
                print("INJECTING RANDOMNESS...")
                mutation_retries += 1
                last_improvement_generation = curr_generation_num
                for member in self.population:
                    member.mutate(mutation_rate * mutation_retry_scale)

            # Get total fitness for the epoch
            print(f"Generation {curr_generation_num}:")
            total_fitness_this_epoch = 0
            best_fitness = -np.inf
            fitnesses = []
            for i, individual in enumerate(self.population):
                # Get statistics for epoch
                fitness = individual.get_fitness()
                fitnesses.append((i, fitness))
                total_fitness_this_epoch += fitness
                if fitness > best_fitness:
                    best_fitness = fitness
                if fitness > max_fitness:
                    max_fitness = fitness
                if total_fitness_this_epoch > max_total_fitness:
                    max_total_fitness = total_fitness_this_epoch
                    last_improvement_generation = curr_generation_num

            # Append to statistics array
            total_fitnesses_per_generation.append(total_fitness_this_epoch)
            best_fitness_per_generation.append(best_fitness)
            max_fitness_over_generations.append(max_fitness)

            # Reproduce
            children = []
            for i in tqdm(range(num_offspring // 2), desc="Reproduction Process: ", ncols=150):
                # Get parents
                parent_1_index, parent_2_index = self._tournament_selection(fitnesses, n_t)
                parent_1 = self.population[parent_1_index]
                parent_2 = self.population[parent_2_index]

                # Generate crossover mask and perform crossover
                crossover_mask = np.random.rand(3, 3) > 0.5
                child_1, child_2 = self._crossover(parent_1, parent_2, crossover_mask)

                # Perform mutation according to mutation rate
                child_1.mutate(mutation_rate)
                child_2.mutate(mutation_rate)

                # Add to children array
                children.extend([child_1, child_2])

            # Select for next generation - using a Steady State Genetic Algorithm
            next_generation = []
            candidates: list[Sudoku] = children
            candidates.extend(self.population.copy())

            # Get fitnesses for candidates
            candidates_fitnesses = []
            for i, candidate in enumerate(candidates):
                fitness = candidate.get_fitness()
                candidates_fitnesses.append((i, fitness))

            # Do tournamenent selection to select next population
            for i in tqdm(range(len(self.population) // 2), desc="Next Generation Selection: ", ncols=150):
                # Get candidate indices
                candidate_1_index, candidate_2_index = self._tournament_selection(candidates_fitnesses, n_t)
                # Get candidates - no need to pop from array because of the first index of the tuple in the fitnesses array containing the candidates index
                candidate_1 = candidates[candidate_1_index]
                candidate_2 = candidates[candidate_2_index]

                # Remove from fitnesses array to prevent the same Grids being added twice
                # Remove first fitness using binary search
                # Made with assistance from https://en.wikipedia.org/wiki/Binary_search
                l = 0
                r = len(candidates_fitnesses) - 1
                index = -1
                while l <= r:
                    m = l + int(math.floor((r - l) / 2))
                    if candidates_fitnesses[m][0] < candidate_1_index:
                        l = m + 1
                    elif candidates_fitnesses[m][0] > candidate_1_index:
                        r = m - 1
                    else:
                        index = m
                        break
                if index == -1:
                    raise Exception("Error: could not find fitness to remove")
                candidates_fitnesses.pop(index)
                # Remove second fitness using binary search
                # Made with assistance from https://en.wikipedia.org/wiki/Binary_search
                l = 0
                r = len(candidates_fitnesses) - 1
                index = -1
                while l <= r:
                    m = l + int(math.floor((r - l) / 2))
                    if candidates_fitnesses[m][0] < candidate_2_index:
                        l = m + 1
                    elif candidates_fitnesses[m][0] > candidate_2_index:
                        r = m - 1
                    else:
                        index = m
                        break
                if index == -1:
                    raise Exception("Error: could not find fitness to remove")
                candidates_fitnesses.pop(index)

                # Add to next generation
                next_generation.extend([candidate_1, candidate_2])

            # Set population to next generation
            self.population = next_generation
            curr_generation_num += 1
            print(f"Total fitness: {total_fitness_this_epoch}")
            print(f"Best fitness: {best_fitness}/{self.max_possible_fitness}")
            print(f"Max fitness: {max_fitness}/{self.max_possible_fitness}\n")

        # Write stats to CSV
        if output_folder.strip():
            # Get x values for x axis for graphs and stats
            x = list(range(len(best_fitness_per_generation)))

            # Display graph for total fitness per generation
            plt.plot(x, total_fitnesses_per_generation)
            plt.xlabel("Epochs")
            plt.ylabel("Total Fitness")
            plt.title("Overall Population Fitness Per Epoch")
            plt.savefig(os.path.join(output_folder, 'total_fitness_per_generation.png'))

            # Display graph for best fitness per generation
            plt.plot(x, best_fitness_per_generation)
            plt.xlabel("Epochs")
            plt.ylabel("Fitness Score")
            plt.title("Best Fitness Score For Each Generation")
            plt.savefig(os.path.join(output_folder, 'best_fitness_per_generation.png'))

            # Display graph for max fitness encountered up to and including each generation
            plt.plot(x, max_fitness_over_generations)
            plt.xlabel("Epochs")
            plt.ylabel("Fitness Score")
            plt.title("Best Fitness Score Encountered Up To Each Epoch")
            plt.savefig(os.path.join(output_folder, 'max_fitness_over_generations.png'))
            
            # Write total fitnesses to CSV
            totals_dict = {
                'generations': x,
                'total_fitness': total_fitnesses_per_generation
            }
            df = pd.DataFrame(totals_dict)
            df.to_csv(os.path.join(output_folder, 'total_fitness_per_generation.csv'))

            # Write best fitness per generation to CSV
            best_fitness_dict = {
                'generations': x,
                'total_fitness': best_fitness_per_generation
            }
            df = pd.DataFrame(best_fitness_dict)
            df.to_csv(os.path.join(output_folder, 'best_fitness_per_generation.csv'))

            # Write max fitness over generations to CSV
            max_fitness_dict = {
                'generations': x,
                'total_fitness': max_fitness_over_generations
            }
            df = pd.DataFrame(max_fitness_dict)
            df.to_csv(os.path.join(output_folder, 'max_fitness_over_generations.csv'))



        # Return solution (if one was found), else return None
        if max_fitness == self.max_possible_fitness:
            for member in self.population:
                if member.get_fitness() == max_fitness:
                    self._solution = member
                    return

    def _tournament_selection(self, fitnesses: list[tuple[int,float]], n_t: int) -> tuple[int, int]:
        # Select tournament candidates and sort them according to fitness
        tournament_candidates = random.sample(fitnesses, n_t)
        tournament_candidates = sorted(tournament_candidates, key=lambda index: index[1], reverse=True)

        # Return the indices in the fitnesses
        return tournament_candidates[0][0], tournament_candidates[1][0]
    
    @abstractmethod
    def _crossover(self, sudoku_1: Sudoku, sudoku_2: Sudoku, crossover_mask: list[list[bool]]) -> tuple[Sudoku, Sudoku]:
        pass


class StandardSudokuEvolutionaryAgent(SudokuEvolutionaryAgent):
    # CONSTRUCTOR
    def __init__(self, grid: np.ndarray, population_size:int = 8100):
        # Initialize super class
        super().__init__(population_size)

        # Get max fitness
        self.max_possible_fitness = len(grid) * len(grid) * 2 # for 9x9, max fitness is 162

        # Instantiate grids
        print("Initializing population...")
        for i in range(self._population_size):
            self.population.append(StandardSudoku(grid.copy()))

    # Made with help from Mantere and Koljonen (2006) - https://www.researchgate.net/profile/Kim-Viljanen/publication/228840763_New_Developments_in_Artificial_Intelligence_and_the_Semantic_Web/links/09e4150a2d2cbb80ff000000/New-Developments-in-Artificial-Intelligence-and-the-Semantic-Web.pdf#page=91
    def _crossover(self, sudoku_1: StandardSudoku, sudoku_2: StandardSudoku, crossover_mask: list[list[bool]]) -> tuple[StandardSudoku, StandardSudoku]:
        # For crossover between 2 members of the population, swap blocks in the same block row and block column
        child_1 = deepcopy(sudoku_1)
        child_2 = deepcopy(sudoku_2)
        for block_row in range(len(crossover_mask)):
            for block_col in range(len(crossover_mask[block_row])):
                if crossover_mask[block_row][block_col]:
                    grid_vals_1 = child_1.get_block_values(block_row, block_col)[0].copy()
                    grid_vals_2 = child_2.get_block_values(block_row, block_col)[0].copy()
                    child_1.set_block_values(block_row, block_col, grid_vals_2)
                    child_2.set_block_values(block_row, block_col, grid_vals_1)
        return child_1, child_2
    


class KillerSudokuEvolutionaryAgent(SudokuEvolutionaryAgent):
    # CONSTRUCTOR
    def __init__(
            self, 
            grid: np.ndarray,
            cages: list[Cage], 
            population_size:int = 8100
        ):
        # Initialize super class
        super().__init__(population_size)

        # Get max fitness
        self.max_possible_fitness = len(grid) * len(grid) * 2 + len(cages)

        # Check if cages are valid
        cage_test_sudoku = KillerSudoku(np.zeros_like(grid), cages, len(grid))
        if not cage_test_sudoku.is_valid_cages():
            raise ValueError("Error: invalid cages provided")

        # Instantiate grids
        print("Initializing population...")
        self.population = [cage_test_sudoku]
        for i in range(1, self._population_size):
            self.population.append(KillerSudoku(np.zeros_like(grid), cages, len(grid)))

    # Made with help from Mantere and Koljonen (2006) - https://www.researchgate.net/profile/Kim-Viljanen/publication/228840763_New_Developments_in_Artificial_Intelligence_and_the_Semantic_Web/links/09e4150a2d2cbb80ff000000/New-Developments-in-Artificial-Intelligence-and-the-Semantic-Web.pdf#page=91
    def _crossover(self, sudoku_1: Sudoku, sudoku_2: Sudoku, crossover_mask: list[list[bool]]) -> tuple[StandardSudoku, StandardSudoku]:
        # For crossover between 2 members of the population, swap blocks in the same block row and block column
        child_1 = deepcopy(sudoku_1)
        child_2 = deepcopy(sudoku_2)
        for block_row in range(len(crossover_mask)):
            for block_col in range(len(crossover_mask[block_row])):
                if crossover_mask[block_row][block_col]:
                    grid_vals_1 = child_1.get_block_values(block_row, block_col).copy()
                    grid_vals_2 = child_2.get_block_values(block_row, block_col).copy()
                    child_1.set_block_values(block_row, block_col, grid_vals_2)
                    child_2.set_block_values(block_row, block_col, grid_vals_1)
        return child_1, child_2
    


class SudokuXEvolutionaryAgent(SudokuEvolutionaryAgent):
    # CONSTRUCTOR
    def __init__(self, grid: np.ndarray, population_size:int=8100):
        # Intialize super class
        super().__init__(population_size)

        # Get max fitness
        self.max_possible_fitness = (2 * len(grid) * len(grid)) + (2 * len(grid))

        # Instantiate grids
        print("Initializing population...")
        for i in range(self._population_size):
            self.population.append(SudokuX(grid.copy()))

    # Made with help from Mantere and Koljonen (2006) - https://www.researchgate.net/profile/Kim-Viljanen/publication/228840763_New_Developments_in_Artificial_Intelligence_and_the_Semantic_Web/links/09e4150a2d2cbb80ff000000/New-Developments-in-Artificial-Intelligence-and-the-Semantic-Web.pdf#page=91
    def _crossover(self, sudoku_1: StandardSudoku, sudoku_2: StandardSudoku, crossover_mask: list[list[bool]]) -> tuple[StandardSudoku, StandardSudoku]:
        # For crossover between 2 members of the population, swap blocks in the same block row and block column
        child_1 = deepcopy(sudoku_1)
        child_2 = deepcopy(sudoku_2)
        for block_row in range(len(crossover_mask)):
            for block_col in range(len(crossover_mask[block_row])):
                if crossover_mask[block_row][block_col]:
                    grid_vals_1 = child_1.get_block_values(block_row, block_col)[0].copy()
                    grid_vals_2 = child_2.get_block_values(block_row, block_col)[0].copy()
                    child_1.set_block_values(block_row, block_col, grid_vals_2)
                    child_2.set_block_values(block_row, block_col, grid_vals_1)
        return child_1, child_2
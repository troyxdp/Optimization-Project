import os
import random
from copy import deepcopy
import math

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from classes.grid import Grid

class EvolutionaryAgent():

    def __init__(self, grid: np.ndarray):
        # Get max fitness
        self.max_possible_fitness = len(grid) * len(grid) * 2 # for 9x9, max fitness is 162

        # Instantiate grids
        self.population : list[Grid] = []
        for i in range(len(grid) * len(grid) * 100):
            self.population.append(Grid(grid.copy()))

        # Initialize grids
        # Initialize each block
        for i in range(int(np.sqrt(len(grid)))):
            for j in range(int(np.sqrt(len(grid)))):
                # Get values that can be changed in the block
                grid_vals, immutabilities = self.population[0].get_block_values(i, j)
                unavailable_values = []
                for val, immutable in zip(grid_vals, immutabilities):
                    if immutable:
                        unavailable_values.append(val)
                available_values = [i for i in range(1, len(grid) + 1) if i not in unavailable_values]

                # Initialize the block for each member of the population
                for sudoku in self.population:
                    # Shuffle the random values
                    random.shuffle(available_values)

                    # Set the block to contain these values
                    values_flat = np.zeros(len(grid))
                    available_values_counter = 0
                    for k in range(len(values_flat)):
                        if immutabilities[k] == 0:
                            values_flat[k] = available_values[available_values_counter]
                            available_values_counter += 1
                        else:
                            values_flat[k] = grid_vals[k]

                    # Set block inside grid to have these values
                    sudoku.set_block_values(i, j, np.array(values_flat))

    # Made with assistance from Dukkipati et al. (2004) - https://arxiv.org/pdf/cs/0408055
    def run_evolutionary_algorithm(self, mutation_rate:float=0.02, no_improvement_max_generations:int=100, num_reproductions_per_epoch:int=4500, n_t:int=16, n_w: int=2):
        print(f"Using a mutation rate of {mutation_rate}...")
        max_fitness = 0
        curr_generation = 0
        last_improvement_generation = 0
        num_epochs = 0
        fitnesses_per_epoch = []
        best_fitness_per_epoch = []
        while not max_fitness == self.max_possible_fitness and curr_generation - last_improvement_generation < no_improvement_max_generations:
            # Get total fitness for the epoch
            total_fitness_this_epoch = 0
            best_fitness = -np.inf
            fitnesses = np.zeros(len(self.population))
            for i, individual in enumerate(self.population):
                # Get statistics for epoch
                fitness = individual.get_fitness()
                fitnesses[i] = fitness
                total_fitness_this_epoch += fitness
                if fitness > best_fitness:
                    best_fitness = fitness
                if fitness > max_fitness:
                    max_fitness = fitness
            
            # Append to statistics array
            fitnesses_per_epoch.append(total_fitness_this_epoch)
            best_fitness_per_epoch.append(best_fitness)

            # Reproduce
            children = []
            for i in range(num_reproductions_per_epoch):
                # Get parent 
                parent_1_index, parent_2_index = self._tournament_selection(fitnesses, n_t, n_w)
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
            candidates: list[Grid] = children.extend(self.population.copy())

            # Get fitnesses for candidates
            candidates_fitnesses = []
            for candidate in candidates:
                fitness = candidate.get_fitness()
                candidates_fitnesses.append(fitness)

            # Do tournamenent selection to select next population
            for i in range(len(self.population)):
                # Get candidate indices
                candidate_1_index, candidate_2_index = self._tournament_selection(fitnesses, n_t, n_w)
                if candidate_2_index > candidate_1_index:
                    # Decrease candidate 2 index if it is after candidate 1 because of the pop() call moving subsequent elements one index lower
                    candidate_2_index -= 1 

                # Remove from candidates array to avoid adding to new generation twice
                candidate_1 = candidates.pop(candidate_1_index)
                candidate_2 = candidates.pop(candidate_2_index)

                # Remove from fitnesses array
                candidates_fitnesses.pop(candidate_1_index)
                candidates_fitnesses.pop(candidate_2_index)

    def _tournament_selection(self, fitnesses: np.ndarray, n_t: int, n_w: int) -> tuple[int, int]:
        ...
                
    # Made with help from Mantere and Koljonen (2006) - https://www.researchgate.net/profile/Kim-Viljanen/publication/228840763_New_Developments_in_Artificial_Intelligence_and_the_Semantic_Web/links/09e4150a2d2cbb80ff000000/New-Developments-in-Artificial-Intelligence-and-the-Semantic-Web.pdf#page=91
    def _crossover(self, grid_1: Grid, grid_2: Grid, crossover_mask: list[list[bool]]) -> tuple[Grid, Grid]:
        # For crossover between 2 members of the population, swap blocks in the same block row and block column
        child_1 = deepcopy(grid_1)
        child_2 = deepcopy(grid_2)
        for block_row in range(len(crossover_mask)):
            for block_col in range(len(crossover_mask[block_row])):
                if crossover_mask[block_row][block_col]:
                    grid_vals_1 = child_1.get_block_values(block_row, block_col)[0].copy()
                    grid_vals_2 = child_2.get_block_values(block_row, block_col)[0].copy()
                    child_1.set_block_values(block_row, block_col, grid_vals_2)
                    child_2.set_block_values(block_row, block_col, grid_vals_1)
        return child_1, child_2

    def _get_g_k(self, g0: float, k: int, alpha: float):
        sigma = 0
        for i in range(1, k + 2):
            sigma += 1 / math.pow(i, alpha)
        return g0 * sigma
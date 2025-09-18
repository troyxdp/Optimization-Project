import os
import random
from copy import deepcopy
import math

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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
    def run_evolutionary_algorithm(
            self, 
            mutation_rate:float=0.02, 
            no_improvement_max_generations:int=300, 
            num_offspring:int=4500, 
            n_t:int=12, 
            output_folder:str='/home/troyxdp/Documents/University Work/Optimization/Project/statistics'
        ):
        # Initialize values
        max_fitness = 0
        last_improvement_generation = 0
        curr_generation_num = 0
        total_fitnesses_per_epoch = []
        max_fitness_over_epochs = []
        best_fitness_per_epoch = []

        # Start evolutionary process
        print("Starting evolution...\n")
        while not max_fitness == self.max_possible_fitness and curr_generation_num - last_improvement_generation < no_improvement_max_generations:
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
                    last_improvement_generation = curr_generation_num
                    max_fitness = fitness
            
            # Append to statistics array
            total_fitnesses_per_epoch.append(total_fitness_this_epoch)
            best_fitness_per_epoch.append(best_fitness)
            max_fitness_over_epochs.append(max_fitness)

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
            candidates: list[Grid] = children
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
            print(f"Best fitness: {best_fitness}")
            print(f"Max fitness: {max_fitness}\n")

        # Get x values for x axis for graphs
        x = list(range(len(best_fitness_per_epoch)))

        # Display graph for overall fitness per epoch
        plt.plot(x, total_fitnesses_per_epoch)
        plt.xlabel("Epochs")
        plt.ylabel("Total Fitness")
        plt.title("Overall Population Fitness Per Epoch")
        plt.show()

        # Write overall fitness stats to CSV

        # Display graph for overall fitness per epoch
        plt.plot(x, best_fitness_per_epoch)
        plt.xlabel("Epochs")
        plt.ylabel("Fitness Score")
        plt.title("Best Fitness Score For Each Generation")
        plt.show()

        # Display graph for overall fitness per epoch
        plt.plot(x, max_fitness_over_epochs)
        plt.xlabel("Epochs")
        plt.ylabel("Fitness Score")
        plt.title("Best Fitness Score Encountered Up To Each Epoch")
        plt.show()

        # Return solution (if one was found)
        if max_fitness == self.max_possible_fitness:
            for member in self.population:
                if member.get_fitness() == max_fitness:
                    return member
        return None

    def _tournament_selection(self, fitnesses: list[tuple[int,float]], n_t: int) -> tuple[int, int]:
        # Select tournament candidates and sort them according to fitness
        tournament_candidates = random.sample(fitnesses, n_t)
        tournament_candidates = sorted(tournament_candidates, key=lambda index: index[1], reverse=True)

        # Return the indices in the fitnesses
        return tournament_candidates[0][0], tournament_candidates[1][0]

                
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
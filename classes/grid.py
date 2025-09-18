import os
import random

import numpy as np

class Grid():

    def __init__(self, grid: np.ndarray):
        self.max_fitness_value = len(grid) * len(grid) * 2
        self.grid = grid
        self.is_immutable = np.zeros_like(grid)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] != 0:
                    self.is_immutable[i][j] = 1
    

    # GETTER METHODS
    def get_fitness(self) -> int:
        # Check rows are valid
        num_valid_digits_in_rows = 0
        for i in range(len(self.grid)):
            is_valid_row = True
            present_numbers = np.zeros(len(self.grid[i]))
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != 0:
                    if present_numbers[self.grid[i][j] - 1] == 0:
                        present_numbers[self.grid[i][j] - 1] = 1
                        num_valid_digits_in_rows += 1

        # Check valid columns
        num_valid_digits_in_cols = 0
        for i in range(len(self.grid)):
            is_valid_row = True
            present_numbers = np.zeros(len(self.grid[i]))
            for j in range(len(self.grid[i])):
                if self.grid[j][i] != 0:
                    if present_numbers[self.grid[j][i] - 1] == 0:
                        present_numbers[self.grid[j][i] - 1] = 1
                        num_valid_digits_in_cols += 1

        return num_valid_digits_in_rows + num_valid_digits_in_cols

    def get_block_values(self, block_row: int, block_col: int) -> np.ndarray:
        # Get starting row and column values and end row and column values
        start_row = int(np.sqrt(len(self.grid)) * block_row)
        start_col = int(np.sqrt(len(self.grid)) * block_col)
        end_row = int(start_row + np.sqrt(len(self.grid)))
        end_col = int(start_col + np.sqrt(len(self.grid)))

        # Get cell values and if they are immutable
        values_flat = np.zeros(len(self.grid))
        values_counter = 0
        is_immutable_flat = np.zeros(len(self.grid))
        is_immutable_counter = 0
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                values_flat[values_counter] = self.grid[i][j]
                values_counter += 1
                is_immutable_flat[is_immutable_counter] = self.is_immutable[i][j]
                is_immutable_counter += 1

        # Return values
        return values_flat, is_immutable_flat
    

    # SETTER METHODS
    def set_block_values(self, block_row: int, block_col: int, values_flat: np.ndarray):
        # Get starting row and column values and end row and column values
        start_row = int(np.sqrt(len(self.grid)) * block_row)
        start_col = int(np.sqrt(len(self.grid)) * block_col)
        end_row = int(start_row + np.sqrt(len(self.grid)))
        end_col = int(start_col + np.sqrt(len(self.grid)))

        # Set values
        values_counter = 0
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                self.grid[i][j] = values_flat[values_counter]
                values_counter += 1


    # FUNCTIONAL METHODS
    def mutate(self, mutation_rate:float=0.02):
        # Iterate through blocks
        num_block_rows = int(np.sqrt(len(self)))
        num_block_cols = num_block_rows
        for row in range(num_block_rows):
            for col in range(num_block_cols):
                r = np.random.rand()
                if r < mutation_rate:
                    self._mutate_block(row, col)

    def _mutate_block(self, block_row: int, block_col: int):
        # Get available values
        grid_vals, immutabilities = self.get_block_values(block_row, block_col)
        unavailable_values = []
        for val, immutable in zip(grid_vals, immutabilities):
            if immutable:
                unavailable_values.append(val)
        available_values = [i for i in grid_vals if i not in unavailable_values]

        if len(available_values) > 1:
            # Randomly swap 2 available values
            # Select the indexes of available values to swap
            available_swap_indexes = list(range(len(available_values)))
            swap_index_1 = random.choice(available_swap_indexes)
            available_swap_indexes.pop(swap_index_1)
            swap_index_2 = random.choice(available_swap_indexes)

            # Swap these values
            temp = available_values[swap_index_1]
            available_values[swap_index_1] = available_values[swap_index_2]
            available_values[swap_index_2] = temp

            # Set the block to contain these values
            values_flat = np.zeros(len(grid_vals))
            available_values_counter = 0
            for k in range(len(values_flat)):
                if immutabilities[k] == 0:
                    values_flat[k] = available_values[available_values_counter]
                    available_values_counter += 1
                else:
                    values_flat[k] = grid_vals[k]

            # Set block inside grid to have these values
            self.set_block_values(block_row, block_col, np.array(values_flat))


    # DUNDER METHODS
    def __str__(self):
        to_ret = ''
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                to_ret += f'{self.grid[i][j]} '
            to_ret = to_ret[:-1] + '\n'
        to_ret = to_ret[:-1]
        return to_ret
    
    def __len__(self):
        return len(self.grid)

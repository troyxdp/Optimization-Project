import random
from abc import ABC, abstractmethod

import numpy as np

class Sudoku(ABC):
    # CONSTRUCTOR
    def __init__(
        self,
        grid:np.ndarray
    ):
        self._grid = grid

    # GETTER METHODS
    def get_fitness(self) -> int:
        # Check rows are valid
        num_valid_digits_in_rows = 0
        for i in range(len(self._grid)):
            present_numbers = np.zeros(len(self._grid[i]))
            for j in range(len(self._grid[i])):
                if self._grid[i][j] != 0:
                    if present_numbers[self._grid[i][j] - 1] == 0:
                        present_numbers[self._grid[i][j] - 1] = 1
                        num_valid_digits_in_rows += 1

        # Check valid columns
        num_valid_digits_in_cols = 0
        for i in range(len(self._grid)):
            present_numbers = np.zeros(len(self._grid[i]))
            for j in range(len(self._grid[i])):
                if self._grid[j][i] != 0:
                    if present_numbers[self._grid[j][i] - 1] == 0:
                        present_numbers[self._grid[j][i] - 1] = 1
                        num_valid_digits_in_cols += 1

        return num_valid_digits_in_rows + num_valid_digits_in_cols
    
    @abstractmethod
    def get_block_values(self, block_row: int, block_col: int) -> np.ndarray:
        pass

    # SETTER METHODS
    def set_block_values(self, block_row: int, block_col: int, values_flat: np.ndarray):
        # Get starting row and column values and end row and column values
        start_row = int(np.sqrt(len(self._grid)) * block_row)
        start_col = int(np.sqrt(len(self._grid)) * block_col)
        end_row = int(start_row + np.sqrt(len(self._grid)))
        end_col = int(start_col + np.sqrt(len(self._grid)))

        # Set values
        values_counter = 0
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                self._grid[i][j] = values_flat[values_counter]
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

    @abstractmethod
    def _mutate_block(self, block_row: int, block_col: int):
       pass
    
    # DUNDER METHODS
    def __str__(self):
        to_ret = ''
        for i in range(len(self._grid)):
            for j in range(len(self._grid[i])):
                to_ret += f'{self._grid[i][j]} '
            to_ret = to_ret[:-1] + '\n'
        to_ret = to_ret[:-1]
        return to_ret
    
    def __len__(self):
        return len(self._grid)



class StandardSudoku(Sudoku):
    # CONSTRUCTOR
    def __init__(self, grid: np.ndarray):
        # Initialize super class values
        super().__init__(grid) # this essentially just initializes the grid

        # Set the max fitness value
        self._max_fitness_value = len(grid) * len(grid) * 2

        # Set the values that are immutable
        self._is_immutable = np.zeros_like(grid)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] != 0:
                    self._is_immutable[i][j] = 1

        # Initialize each block
        for i in range(int(np.sqrt(len(grid)))):
            for j in range(int(np.sqrt(len(grid)))):
                # Get values that can be changed in the block
                grid_vals, immutabilities = self.get_block_values(i, j)
                unavailable_values = []
                for val, immutable in zip(grid_vals, immutabilities):
                    if immutable:
                        unavailable_values.append(val) # This works because the value for self._grid is already set when calling the super class constructor
                available_values = [i for i in range(1, len(grid) + 1) if i not in unavailable_values]

                # Initialize the block for each member of the population
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
                self.set_block_values(i, j, np.array(values_flat))

    # GETTER METHODS
    def get_block_values(self, block_row: int, block_col: int) -> np.ndarray:
        # Get starting row and column values and end row and column values
        start_row = int(np.sqrt(len(self._grid)) * block_row)
        start_col = int(np.sqrt(len(self._grid)) * block_col)
        end_row = int(start_row + np.sqrt(len(self._grid)))
        end_col = int(start_col + np.sqrt(len(self._grid)))

        # Get cell values and if they are immutable
        values_flat = np.zeros(len(self._grid))
        values_counter = 0
        is_immutable_flat = np.zeros(len(self._grid))
        is_immutable_counter = 0
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                values_flat[values_counter] = self._grid[i][j]
                values_counter += 1
                is_immutable_flat[is_immutable_counter] = self._is_immutable[i][j]
                is_immutable_counter += 1

        # Return values
        return values_flat, is_immutable_flat

    # SETTER METHODS
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



class Cage():
    # CONSTRUCTOR
    def __init__(
        self, 
        cells: list[tuple[int]], 
        correct_sum: int
    ):
        self._cells = cells
        self._correct_sum = correct_sum

    # GETTER METHODS
    def get_correct_sum(self):
        return self._correct_sum

    def get_sum(self, grid:np.ndarray):
        sigma = 0
        for cell in self._cells:
            sigma += grid[cell[0]][cell[1]]
        return sigma
    
    # DUNDER METHODS
    def __contains__(self, cell_tuple:tuple[int,int]) -> bool:
        for cell in self._cells:
            if cell == cell_tuple:
                return True
        return False



class KillerSudoku(Sudoku):
    # CONSTRUCTOR
    def __init__(
        self,
        grid:np.ndarray,
        cages:list[Cage],
        grid_size:int=9
    ):
        # Check if cages are valid
        if not self.is_valid_cages(cages):
            raise ValueError("Error: invalid cages provided")
        
        # Initialize super class values
        super().__init__(grid) # this essentially just instantiates the grid

        # Get max fitness value, which is the max fitness value of a normal sudoku plus the number of cages that are valid
        self._max_fitness_value = 2 * grid_size * grid_size + len(cages)

        # Initialize the grid by creating valid blocks
        for row in range(int(np.sqrt(grid_size))):
            for col in range(int(np.sqrt(grid_size))):
                # Get block
                row_start = row * int(np.sqrt(grid_size))
                row_end = row_start + int(np.sqrt(grid_size))
                col_start = col * int(np.sqrt(grid_size))
                col_end = col_start + int(np.sqrt(grid_size))

                # Set block values
                block_vals = list(range(1, grid_size + 1))
                random.shuffle(block_vals)
                idx = 0
                for block_row in range(row_start, row_end):
                    for block_col in range(col_start, col_end):
                        self._grid[block_row, block_col] = block_vals[idx]
                        idx += 1

        # Initialize cages
        self._cages: list[Cage] = cages
    
    # GETTER METHODS
    def get_fitness(self) -> int:
        row_col_fitness = super().get_fitness()

        # Get number of valid cages
        num_valid_cages = 0
        for cage in self._cages:
            cage_sum = cage.get_sum(self._grid)
            if cage_sum == cage.get_correct_sum():
                num_valid_cages += 1

        # Return fitness value = number of unique values in each row + number of unique values in each column + number of valid cages
        return row_col_fitness + num_valid_cages

    def get_block_values(self, block_row: int, block_col: int) -> np.ndarray:
        # Get starting row and column values and end row and column values
        start_row = int(np.sqrt(len(self._grid)) * block_row)
        start_col = int(np.sqrt(len(self._grid)) * block_col)
        end_row = int(start_row + np.sqrt(len(self._grid)))
        end_col = int(start_col + np.sqrt(len(self._grid)))

        # Get cell values and if they are immutable
        values_flat = np.zeros(len(self._grid))
        values_counter = 0
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                values_flat[values_counter] = self._grid[i][j]
                values_counter += 1

        # Return values
        return values_flat

    # SETTER METHODS
    def _mutate_block(self, block_row: int, block_col: int):
        # Get the values in the block
        block_vals = self.get_block_values(block_row, block_col)
        block_vals = block_vals.tolist()
        
        # Get indices to swap
        available_swap_indexes = list(range(len(self._grid)))
        swap_index_1 = random.choice(available_swap_indexes)
        available_swap_indexes.pop(swap_index_1)
        swap_index_2 = random.choice(available_swap_indexes)

        # Swap the indices and update the block
        temp = block_vals[swap_index_1]
        block_vals[swap_index_1] = block_vals[swap_index_2]
        block_vals[swap_index_2] = temp
        self.set_block_values(block_row, block_col, np.array(block_vals))

    # FUNCTIONAL METHODS
    def is_valid_cages(self, cages: list[Cage], grid_size:int=9):
        for row in range(grid_size):
            for col in range(grid_size):
                cell_tuple = (row, col)
                contained_in_cages = False
                for cage in cages:
                    if cell_tuple in cage:
                        contained_in_cages = True
                        break
                if not contained_in_cages:
                    return False
        return True
import os
import subprocess

# Variables being tested
MUTATION_RATES = [0.02, 0.03, 0.06]
TOURNAMENT_SIZES = [8, 12, 16]
POPULATION_SIZES = [8100] # set the number of offspring to be half these values
NUM_EXPERIMENTS_PER_CONFIGURATION = 3

if __name__ == '__main__':
    # Tests on Standard Sudoku
    standard_in_dir = '/home/troyxdp/Documents/University Work/Optimization/Project/test_data/Standard Sudoku'
    standard_out_dir = '/home/troyxdp/Documents/University Work/Optimization/Project/statistics/Standard Sudoku'
    for input_puzzle in sorted(os.listdir(standard_in_dir)):
        # Get puzzle path and subfolder to send results to
        subfolder = os.path.splitext(input_puzzle)[0]
        puzzle_path = os.path.join(standard_in_dir, input_puzzle)

        # Perform experiments for each of the different hyperparameters
        experiment_counter = 1
        for mutation_rate in MUTATION_RATES:
            for n_t in TOURNAMENT_SIZES:
                for pop_size in POPULATION_SIZES:
                    for _ in range(NUM_EXPERIMENTS_PER_CONFIGURATION):
                        # Get output path and create the directory for it if need be
                        output_path = os.path.join(standard_out_dir, subfolder, f'experiment_{experiment_counter}')
                        if not os.path.isdir(output_path):
                            os.makedirs(output_path)
                        # else:
                        #     print(f"Already performed experiment for output folder {output_path}")
                        #     experiment_counter += 1
                        #     continue

                        # Perform experiment
                        subprocess.run([
                            "python3",
                            "main.py",
                            "--sudoku-path", puzzle_path,
                            "--output-folder", output_path,
                            "--nt", str(n_t),
                            "--mutation-rate", str(mutation_rate),
                            "--population-size", str(pop_size),
                            "--num-offspring", str(pop_size // 2)
                        ])

                        # Update experiment_counter
                        experiment_counter += 1

    # Tests on Sudoku X
    x_in_dir = '/home/troyxdp/Documents/University Work/Optimization/Project/test_data/Sudoku X'
    x_out_dir = '/home/troyxdp/Documents/University Work/Optimization/Project/statistics/Sudoku X'
    for input_puzzle in sorted(os.listdir(x_in_dir)):
        # Get puzzle path and subfolder to send results to
        subfolder = os.path.splitext(input_puzzle)[0]
        puzzle_path = os.path.join(x_in_dir, input_puzzle)

        # Perform experiments for each of the different hyperparameters
        experiment_counter = 1
        for mutation_rate in MUTATION_RATES:
            for n_t in TOURNAMENT_SIZES:
                for pop_size in POPULATION_SIZES:
                    for _ in range(NUM_EXPERIMENTS_PER_CONFIGURATION):
                        # Get output path and create the directory for it if need be
                        output_path = os.path.join(x_out_dir, subfolder, f'experiment_{experiment_counter}')
                        if not os.path.isdir(output_path):
                            os.makedirs(output_path)
                        # else:
                        #     print(f"Already performed experiment for output folder {output_path}")
                        #     experiment_counter += 1
                        #     continue

                        # Perform experiment
                        subprocess.run([
                            "python3",
                            "main.py",
                            "--sudoku-x",
                            "--sudoku-path", puzzle_path,
                            "--output-folder", output_path,
                            "--nt", str(n_t),
                            "--mutation-rate", str(mutation_rate),
                            "--population-size", str(pop_size),
                            "--num-offspring", str(pop_size // 2)
                        ])
                        # Update experiment_counter
                        experiment_counter += 1

    # Tests on Killer Sudoku
    killer_in_dir = '/home/troyxdp/Documents/University Work/Optimization/Project/test_data/Killer Sudoku'
    killer_out_dir = '/home/troyxdp/Documents/University Work/Optimization/Project/statistics/Killer Sudoku'
    for input_puzzle in sorted(os.listdir(killer_in_dir)):
        # Get puzzle path and subfolder to send results to
        subfolder = os.path.splitext(input_puzzle)[0]
        puzzle_path = os.path.join(killer_in_dir, input_puzzle)

        # Perform experiments for each of the different hyperparameters
        experiment_counter = 1
        for mutation_rate in MUTATION_RATES:
            for n_t in TOURNAMENT_SIZES:
                for pop_size in POPULATION_SIZES:
                    for _ in range(NUM_EXPERIMENTS_PER_CONFIGURATION):
                        # Get output path and create the directory for it if need be
                        output_path = os.path.join(killer_out_dir, subfolder, f'experiment_{experiment_counter}')
                        if not os.path.isdir(output_path):
                            os.makedirs(output_path)
                        # else:
                        #     print(f"Already performed experiment for output folder {output_path}")
                        #     experiment_counter += 1
                        #     continue

                        # Perform experiment
                        subprocess.run([
                            "python3",
                            "main.py",
                            "--killer",
                            "--sudoku-path", puzzle_path,
                            "--output-folder", output_path,
                            "--nt", str(n_t),
                            "--mutation-rate", str(mutation_rate),
                            "--population-size", str(pop_size),
                            "--num-offspring", str(pop_size // 2)
                        ])

                        # Update experiment_counter
                        experiment_counter += 1
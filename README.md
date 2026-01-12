# Optimization Project
Welcome to my Optimization project for my Honours year! It is a Sudoku solver that makes use of a genetic algorithm. It also solves Killer Sudoku and Sudoku X puzzles. 

To run the project, execute the following command:
```
python main.py
  --sudoku-path <path-to-txt-file>
  [--mutation-rate <float>]
  [--nt <int>]
  [--no-improvement-max-generations <int>]
  [--num-mutation-retries <int>]
  [--mutation-retry-scale <float>]
  [--num-offspring <int>]
  [--output-folder <path-to-folder>]
  [--killer]
  [--sudoku-x]
  [--population-size <int>]
```
For more information on the different arguments, run the following command:
```
python main.py --help
```

The input for standard sudoku and sudoku X puzzles is of the following form:
```
0 0 3 0 2 0 6 0 0
9 0 0 3 0 5 0 0 1
0 0 1 8 0 6 4 0 0
0 0 8 1 0 2 9 0 0
7 0 0 0 0 0 0 0 8
0 0 6 7 0 8 2 0 0
0 0 2 6 0 9 5 0 0
8 0 0 2 0 3 0 0 9
0 0 5 0 1 0 3 0 0
```
where a 0 represent an empty cell. If the puzzle input is not valid, an error will be thrown.

I honestly cannot remember the format for inputing a killer sudoku, but I will check and add it hear when I can.

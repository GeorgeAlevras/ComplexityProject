# Complexity Project #

Author: George Alevras \
Date: 22/02/2021

## Description ##
This was my first of two courseworks representing 45% of the course 'Complexity & Networks' - obtaining a grade of 71.2%. The project involved building an Oslo model - one of the simplest models displaying self-organised criticality (SOC). Algorithms were developed to explore how properties of the model scale with model size, analysing transient and recurring configurations. Finally, moment analysis was applied to demonstrate fundamental aspects of self-organised criticality - scaling and data collapse.

## Organisation ##
The repository contains:
- The report submitted for the coursework `./01531221-ComplexityReport.pdf`
- The file with the main source code `./main.py`
- A file that does logarithmic binning of data `./logbin2020.py`
- A folder with all data object files (Numpy) where data structures are saved `./Numpy Files`
- A folder with all data object files (Numpy) where data structures from previous runs are saved `./Old Files`
- A folder with the plots that are produced from the experiments `./Plots`


## Execute Code: ##
To run the code you need to consider 2 arguments:
    
    -t (task number, type=str) and -e (execute, type=boolean)

The first argument is necessary and identifies which task to run. If the 2nd argument is present then the code will run the Oslo model to produce the data needed for that task. If the 2nd argument is not present then it will use the already-made data to produce the plots for that task. It takes a lot of time (5-60 minutes to produce the data, so would advise to just plot, aka do NOT includethe -e argument).

### Example commands: ###
    
    1. python3 main.py -t 2e
    2. python3 main.py -t 2g -e
    3. python3 main.py -h

## Code Structure: ##

The code contains 4 main methods that define the Oslo model and one method for each task, which are listed in the project script 'CandNProject.pdf'.

### The 4 main methods are: ###

    1. threshold_prob: Returns random threshold value by determining the probability for each
    2. initialise: Initialises the sites of a system of given size L and threshold probability p
    3. update_slopes: Updates all the threshold values of the system
    4. drive_and_relax:Keeps driving and relaxing a system achieving a final stable configuration until 
        a given number of grains have been added to the system, with threshold probability p

Then there are methods named 'task_1', 'task_2a' etc. one for each task

# Necessary Conditions: #
    1. Saved .npy files must be in same directory as the main.py file
    2. logbin_2020.py file must be in same directory as the main.py file
    3. -t argument must always be given: type 'python3 main.py -h' to get help

*Enjoy the code!*
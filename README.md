# ComplexityProject #

Author: Georgios Alevras \
Date: 18/02/2021


Python Version: 3.8.2
### Dependencies: ###
- Numpy: v1.19.1
- Matplotlib: v3.3.1
- Scipy: v1.5.2
- os
- collections
- argparse
- logbin_2020


## Execute Code: ##
To run the code you need to consider 2 arguments: \
    
    - -t (task number, type=str) and -e (execute, type=boolean)

The first argument is necessary and identifies which task to run. If the 2nd argument is present then the code will run the Oslo model to produce the data needed for that task. If the 2nd argument is not present then it will use the already-made data to produce the plots for that task. \

Example commands: \
    
    1. python3 main.py -t 2e
    2. python3 main.py -t 2g -e
    3. python3 main.py -h

## Code Structure: ##

The code contains 4 main methods that define the Oslo model, and another method for each task, as they are listed in the project script 'CandNProject.pdf'. \

    1. Custom PDF used for threshold
    2. Initialisation method to start model for given system size
    3. Drive and Relax method to run system for given No. of grains
    4. Task_1 method for Task 1: to perform tests on model - ensuring it works
    5. Task_2 method for Task 2

# Necessary Conditions: #
    1. Saved .npy files must be in same directory as main.py file
    2. logbin_2020.py file must be in same directory as main.py file
    3. -t argument must always be given, type 'python3 main.py -h' to get help

*Enjoy the code!*
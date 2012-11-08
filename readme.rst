Genetic Algorithm implementation in Python for the Traveling Salesman Problem
==============================================================================

The main script is the `ga.py` file which contains base classes for creating
your own Genetic Algorithm (i.e.
`import ga; env = ga.Environment(kind=YourKind)` where `YourKind` is some
class inheriting `IndividualBase`) or for simply running the provided
implementation of the Traveling Salesman Problem.

Usage for the latter:

::

    Usage: ga.py [options]

    Options:
    -h, --help            show this help message and exit
    -g MAX_GENERATIONS, --max-generations=MAX_GENERATIONS
                            Continue till we have reached MAX_GENERATIONS
    -p POPULATION_SIZE, --population-size=POPULATION_SIZE
                            Store POPULATION_SIZE different individuals
    -c CROSSOVER_RATE, --crossover-rate=CROSSOVER_RATE
                            Set crossover probability to CROSSOVER_RATE (between 0
                            and 1)
    -m MUTATION_RATE, --mutation-rate=MUTATION_RATE
                            Set mutation probability to MUTATION_RATE (between 0
                            and 1)
    -e ELITISM, --elitism=ELITISM
                            Enable elitism for the top ELITISM results
    --print-interval=PRINT_INTERVAL
                            Print intermediate results for every PRINT_INTERVAL
                            generations. Use 0 for no intermediate output
    --csv=CSV             Return csv output for easy plotting every CSV
                            generation
    --csv-file=CSV_FILE   Where to write the csv output to (defaults to STDOUT)
    -s SAMPLES, --samples=SAMPLES
                            The amount of samples to use. Very useful with csv
                            output
    --processes=PROCESSES
                            By default the samples are calculated in parallel, you
                            can change the amount of simultaneous processes with
                            PROCESSES

To generate all output automatically there is also a script called
`generate_charts.sh` available which automatically try all kinds of different
values for elitism, population, mutation and crossover.


If you have any questions, feel free to mail me at: `Rick _at_ Fawo _dot_ nl`


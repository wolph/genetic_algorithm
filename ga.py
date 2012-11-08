#!/usr/bin/env python

import sys
import random
import optparse
import multiprocessing

try:
    # Since we like pretty colours we see if the fabulous module is installed.
    # If it is, than we can color-code the cities to present a nice graphical
    # representation of the evolution.
    from fabulous.color import bg256, fg256
except ImportError:
    fg256 = bg256 = lambda color, text: text
    print >>sys.stderr, 'Fabulous not installed, no color output available'

def generate_colours(interval):
    r, g, b = 255, 0, 0
    step = 256 / interval
    out = []
    for g in range(step - 1, 256, step):
        out.append('#%02x%02x%02x' % (r, g, b))

    for r in range(255, 0, -step):
        out.append('#%02x%02x%02x' % (r, g, b))

    for b in range(step - 1, 256, step):
        out.append('#%02x%02x%02x' % (r, g, b))

    for g in range(255, 0, -step):
        out.append('#%02x%02x%02x' % (r, g, b))

    for r in range(step - 1, 256, step):
        out.append('#%02x%02x%02x' % (r, g, b))

    for b in range(255, 0, -step):
        out.append('#%02x%02x%02x' % (r, g, b))

    return out
        
colours = generate_colours(16)

def colour(max, i):
    out = '%02d' % i
    value = i * (len(colours) / max)
    return str(fg256('#000', bg256(colours[value], out)))

def maximize(self, a, b):
    '''Tell the individual to maximize towards the solution'''
    return cmp(b.score, a.score)

def minimize(self, a, b):
    '''Tell the individual to minimize towards the solution'''
    return cmp(a.score, b.score)

def pickpivots(individual):
    '''
    Get two lists of random pivots in the chromosome for the given individual
    '''
    left = random.randrange(1, individual.length - 2)
    right = random.randrange(left, individual.length - 1)
    return left, right


class CrossoverMixin(object):
    def crossover(self, other):
        'A crossover method which returns a list of newly created offspring'
        return []


class TwopointCrossoverMixin(CrossoverMixin):
    '''
    Simple twopoint crossover method, not that suited for TSP since it
    requires a smart/complex repair method which beats the purpose
    
    Would be useful for many other problems and/or for the TSP problem given
    a different encoding
    '''
    def crossover(self, other):
        '''Return a twopoint crossover for the given chromosomes'''
        left, right = pickpivots(self)

        def mate(self, other):
            chromosome = self.chromosome
            chromosome[left:right] = other.chromosome[left:right]
            child = self.copy(chromosome)
            if child.repair:
                child.repair(self, other)
            return child

        return mate(self, other), mate(other, self)


class EdgeRecombinationCrossoverMixin(CrossoverMixin):
    '''
    Implemented as specified here:
    http://en.wikipedia.org/wiki/Edge_recombination_operator
    '''
    def get_adjacency_matrix(self, matrix):
        for i, g in enumerate(self.chromosome):
            connections = matrix.get(g, {})
            # Store the next and previous node as connections
            # We use the modulo operator to do a simple wraparound
            connections[self.chromosome[(i + 1) % self.length]] = None
            connections[self.chromosome[
                (i - 1 + self.length) % self.length]] = None
            matrix[g] = connections

        return matrix

    def get_distance_matrix(self, matrix):
        for from_, cities in matrix.items():
            for to in cities:
                cities[to] = self.env.cities[from_][to]

            matrix[from_] = [k for k, v 
                in sorted(cities.iteritems(), key=lambda (k, v): (v, k))]

        return matrix

    def get_closest_neighbour(self, neighbours, unavailable):
        for neighbour in neighbours:
            if neighbour not in unavailable:
                return neighbour

    def crossover(self, other):
        return self.mate(self, other), self.mate(other, self)

    def mate(*parents):
        self = parents[0]

        # Build the adjacency matrix for the parents
        matrix = {}
        for parent in parents:
            matrix = parent.get_adjacency_matrix(matrix)

        matrix = self.get_distance_matrix(matrix)

        # pick the first node of a random parent
        n = random.choice(parents).chromosome[0]

        k = []
        unavailable = {}
        for _ in range(self.length - 1):
            k.append(n)
            unavailable[n] = True

            # Try to get the closest neighbour from our distance matrix
            neighbour = self.get_closest_neighbour(matrix[n], unavailable)

            # Couldn't find anything, let's try the neighbours of our
            # neighbours
            if neighbour is None:
                neighbours = matrix[n][:]
                random.shuffle(neighbours)
                for i in neighbours:
                    neighbour = self.get_closest_neighbour(
                        matrix[i],
                        unavailable,
                    )
                    if neighbour is not None:
                        break

            # We're really desperate now... let's try any random missing node
            if neighbour is None and len(k) < self.length:
                missing = list(set(self.alleles) - set(k))
                neighbour = random.choice(list(missing))

            assert neighbour is not None, 'Unable to get a full length ' \
                'chromosome: %s' % unavailable.keys()
            n = neighbour

        k.append(n)
        child = self.copy(k)
        return child


class MutatorMixin(object):
    def mutate(self, gene):
        'Mutate the individual'


class RandomMutatorMixin(MutatorMixin):
    '''Mutator which just picks a fully random gene'''
    def mutate(self, gene):
        'Mutate by random mutation'
        self.chromosome[gene] = random.choice(self.alleles)


class RandomSwapMutatorMixin(MutatorMixin):
    '''
    Quite similar to the RandomMutator except that this mutator swaps 2
    genes which makes it suitable for permutation based chromosomes
    '''
    def mutate(self, gene):
        'Mutate by random swap mutation'
        c = self.chromosome
        chromosome = len(c)
        if chromosome > 2:
            index = gene
            while index == gene:
                index = random.randrange(chromosome)

            c[gene], c[index] = c[index], c[gene]

class EvaluatorMixin(object):
    def evaluate(self):
        'Update the score for the individual'


class SumEvaluatorMixin(object):
    '''Simple evaluator which calculates the sum of the chromosome'''
    def evaluate(self):
        self.score = sum(self.chromosome)


class IndividualBase(object):
    length = 30
    alleles = (0,1)
    optimization = minimize

    # Enable a crossover/mutate/repair method by subclassing the
    # Mutator/Crossover/Repair/Evaluate mixins
    crossover = None
    mutate = None
    repair = None
    evaluate = None

    def __init__(self, env, chromosome=None):
        self.env = env
        self.chromosome = chromosome
    
    def set_chromosome(self, chromosome):
        '''
        Set or create a new chromosome. If no chromosome is given, create
        one based on the chromosome length based on the available alleles
        '''
        if not chromosome:
            # By skipping the first allele we can fix the starting point
            # chromosome = self.alleles[1:]
            # random.shuffle(chromosome)
            # chromosome.insert(0, 0)
            chromosome = self.alleles[:]
            random.shuffle(chromosome)
        self._chromosome = chromosome

        # With a new chromosome we need score of 0
        self.score = None
        # Recalculate the score
        self.evaluate()

    def get_chromosome(self):
        '''Return the current chromosome'''
        return self._chromosome

    chromosome = property(get_chromosome, set_chromosome)

    def __repr__(self):
        chromosome = self.chromosome[:]
        chromosome = ' '.join(colour(self.length, i) for i in chromosome)

        'Return the representation of the individual'
        return (u'<%s[%s]: %s>' % (
            self.__class__.__name__,
            self.score,
            chromosome,
        )).encode('utf-8')

    def __cmp__(self, other):
        return self.optimization(self, other)
    
    def copy(self, chromosome=None):
        chromosome = chromosome or self.chromosome
        twin = self.__class__(self.env, chromosome[:])
        return twin


class SelectorMixin(object):
    def select(self):
        'A selection method to return a individual from our population'


class TournamentSelectorMixin(SelectorMixin):
    '''
    The tournament selector creates a tournament with a specific size of
    random individuals from the population.

    After creating the tournament we have a certain probability (default 0.9)
    of selecting the best individual in the tournament. Otherwise we simply
    take a random competitor.
    '''
    def __init__(self, tournament_size=8, tournament_choosebest=0.9, **kwargs):
        self.tournament_size = tournament_size
        self.tournament_choosebest = tournament_choosebest
        super(TournamentSelectorMixin, self).__init__(**kwargs)

    def select(self):
        competitors = [random.choice(self.population)
            for i in range(self.tournament_size)]

        competitors.sort()
        if random.random() < self.tournament_choosebest:
            return competitors[0]
        else:
            return random.choice(competitors[1:])

class Environment(object):
    '''The environment contains the population and manages the generations.'''
    # Subclass this and set this to one of the selectors
    select = None

    def __init__(self, kind, population=None, population_size=100, 
            max_generations=100, crossover_rate=0.90, mutation_rate=0.01,
            print_interval=10, elitism=1, **kwargs):
        self.generation = 0
        self.elitism = elitism
        self.kind = kind
        self.population_size = population_size
        self.population = population
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.print_interval = print_interval

        for individual in self.population:
            individual.evaluate()

    def get_population(self):
        return self._population

    def set_population(self, population):
        if not population:
            population = [self.kind(env=self)
                for individual in range(self.population_size)]
        self._population = population

    population = property(get_population, set_population)

    def run(self):
        for generation in self:
            pass
    
    def __iter__(self):
        '''Evolve until we've reached the max generations'''
        self.generation = 0
        while self.generation < self.max_generations:
            yield self.next()

    def next(self):
        '''Next evolution'''
        self.population.sort()
        self.crossover()
        self.generation += 1
        if self.print_interval and self.generation % self.print_interval == 0:
            print >>sys.stderr, repr(self)

        return self
    
    def crossover(self):
        '''Crossover every individual in the population with the provided
        crossover rate.'''
        next_population = []
        for i in range(self.elitism):
            next_population.append(self.population[i].copy())

        while len(next_population) < self.population_size:
            a = self.select()

            # Depending on the crossover rate, crossover or simply keep our
            # individual
            offspring = [a.copy()]
            if random.random() < self.crossover_rate:
                b = self.select()
                if a != b:
                    offspring = a.crossover(b)

            # Mutate all the individuals and store the scores
            for individual in offspring:
                self.mutate(individual)

                # Evaluate the results for the individual and add it to the
                # list of next populations
                individual.evaluate()
                next_population.append(individual)

        # Keep only the latest results
        self.population = next_population[:self.population_size]
    
    def mutate(self, individual):
        '''Mutate all genes for the given individual with the provided
        mutation rate'''
        for gene in range(individual.length):
            if random.random() < self.mutation_rate:
                individual.mutate(gene)

    @property
    def best(self):
        'individual with best fitness score in population.'
        return self.population[0]

    def __repr__(self):
        # return 'generation %s, %s' % (
        #     self.generation,
        #     self.population,
        # )
        return 'generation %s, best: %s' % (
            self.generation,
            self.best,
        )

   
class TSPEnv(TournamentSelectorMixin, Environment):
    '''The Traveling Salesman Problem environment.

    Uses the tournament selector for selecting crossover individuals and
    generates a distance table for all closests cities from every other city.
    '''
    def __init__(self, **kwargs):
        cities = kwargs['cities']
        kind = kwargs['kind']

        # As alleles we use the indexes of the cities array
        kind.alleles = range(len(cities))
        kind.length = len(cities)
        self.cities = cities
        self.closest = self.get_closest(cities)

        super(TSPEnv, self).__init__(**kwargs)

    def get_closest(self, cities):
        closest = {}
        for distances in cities:
            # Sort the cities by distances
            distances = sorted(enumerate(distances), key=lambda x: x[::-1])
            # Drop the score and keep the index only
            distances = [k for k, _ in distances]
            closest[distances[0]] = distances[1:]

        return closest

class PermutationCrossoverRepairMixin(object):
    '''If the normal (e.g. Two Point Crossover) operators are used than we
    need a smart repair method. Since this wasn't really the way to go we
    replaced this method and the crossover with the Edge Recombination
    Crossover.
    '''
    def repair(self, mother, father):
        chromosome = set(self.chromosome)
        missing = sorted(set(self.alleles) - chromosome)

        if not missing:
            # No need for repair, all chromosomes are there :)
            return

        unique_alleles = {}
        for i, allele in enumerate(self.chromosome):
            if allele in unique_alleles:
                for missing_allele in missing:
                    for close_allele in self.env.closest:
                        if close_allele not in chromosome:
                            unique_alleles[close_allele] = True
                            self.chromosome[i] = close_allele
            else:
                unique_alleles[allele] = True

class TSP(EdgeRecombinationCrossoverMixin, RandomSwapMutatorMixin,
        IndividualBase):
    '''The Traveling Salesman Problem class combines the Edge Recombination
    with a simple Random Swap to provide for a permutation based chromosome.

    The chromosome will be a list of city numbers where the order decides the
    order in which the cities are visited.
    '''
    optimization = minimize

    def evaluate(self):
        score = 0
        cities = self.env.cities

        # calculate the cost by using the chromosome values as indexes for the
        # cities array
        for i in range(len(cities) - 1):
            x, y = self.chromosome[i:i+2]
            score += cities[x][y]

        self.score = score

def get_cities(count):
    '''Get the cities for the given count from the data/<cities>_cities.txt
    file'''
    fh = open('data/%s_cities.txt' % count)

    # The first line contains the amount of cities, not needed with Python
    # lists so we ignore it
    fh.next()

    matrix = []
    for i, distances in enumerate(fh):
        matrix.append([])
        for distance in distances.split():
            matrix[i].append(float(distance))

    return matrix

def run_sample((sample, kwargs)):
    print >>sys.stderr, 'Starting sample %s with %s cities' % (
        sample,
        len(kwargs['cities']),
    )
    data = []
    env = TSPEnv(kind=TSP, **kwargs)
    for generation in env:
        if csv and generation.generation % csv == 0:
            data.append((
                generation.generation,
                sample,
                generation.best.score,
            ))
    data.append(())

    return data

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-g', '--max-generations', type='int', default=1000,
        help='Continue till we have reached MAX_GENERATIONS')
    parser.add_option('-p', '--population-size', type='int', default=10,
        help='Store POPULATION_SIZE different individuals')
    parser.add_option('-c', '--crossover-rate', type='float', default=0.9,
        help='Set crossover probability to CROSSOVER_RATE (between 0 and 1)')
    parser.add_option('-m', '--mutation-rate', type='float', default=0.01,
        help='Set mutation probability to MUTATION_RATE (between 0 and 1)')
    parser.add_option('-e', '--elitism', type='int', default=1,
        help='Enable elitism for the top ELITISM results')
    parser.add_option('--print-interval', type='int', default=100,
        help='Print intermediate results for every PRINT_INTERVAL '
        'generations. Use 0 for no intermediate output')
    parser.add_option('--csv', type='int', default=0,
        help='Return csv output for easy plotting every CSV generation')
    parser.add_option('--csv-file', type='string',
        help='Where to write the csv output to (defaults to STDOUT)')
    parser.add_option('-s', '--samples', type='int', default=1,
        help='The amount of samples to use. Very useful with csv output')
    parser.add_option('--processes', type='int',
        help='By default the samples are calculated in parallel, you can '
            'change the amount of simultaneous processes with PROCESSES')

    options, city_counts = parser.parse_args()
    kwargs = dict(options.__dict__)

    if not city_counts:
        parser.print_help()
        sys.exit(1)

    csv = kwargs['csv']
    samples = kwargs['samples']
    generations = kwargs['max_generations']

    data = []
    # To allow for faster processing we use the Python Multiprocessing module
    # to use all difference CPU's. If needed this can easily be extended to a
    # distributed version where multiple servers are helping with the
    # calculations.
    pool = multiprocessing.Pool(kwargs['processes'])
    for city_count in city_counts:
        cities = get_cities(city_count)
        print 'Processing %d cities' % len(cities)
        kwargs['cities'] = cities

        if samples > 1:
            data += sum(
                pool.map(
                    run_sample,
                    [(sample, kwargs) for sample in range(samples)],
                    chunksize=1,
                ),
                []
            )
        else:
            data += run_sample((0, kwargs))

    if csv:
        if kwargs['csv_file']:
            csv_fh = open(kwargs['csv_file'], 'w')
        else:
            csv_fh = sys.stdout

        for generation in data:
            if generation:
                print >>csv_fh, ';'.join(map(str, generation))
            else:
                print >>csv_fh, ''


# src/autora/theorist/nuts/genetic/evolve.py
import random, math
from .selection import tournament
from .crossover import subtree_crossover
from .mutation import subtree_mutation, tweak_constants

def evolve(pop, consts_list, X, y, config, fit_fn, grow_fn, logger):
    for gen in range(config.generations):
        fitnesses = [fit_fn(ind, X, y, c) for ind, c in zip(pop, consts_list)]
        logger.on_generation(gen, pop, consts_list, fitnesses)

        new_pop, new_consts = [], []
        elite_idxs = sorted(range(len(pop)), key=lambda i: fitnesses[i])[:config.elitism]
        for i in elite_idxs:
            new_pop.append(pop[i])
            new_consts.append(consts_list[i])

        while len(new_pop) < config.population_size:
            if random.random() < config.crossover_rate:
                p1 = tournament(pop, fitnesses, config.tournament_k)
                p2 = tournament(pop, fitnesses, config.tournament_k)
                child = subtree_crossover(p1, p2, config.max_depth)
            else:
                p1 = tournament(pop, fitnesses, config.tournament_k)
                child = subtree_mutation(p1, grow_fn)

            if random.random() < config.mutation_rate:
                child = subtree_mutation(child, grow_fn)

            c = tweak_constants(random.choice(consts_list), config.constant_mutation_sigma)
            new_pop.append(child)
            new_consts.append(c)

        pop, consts_list = new_pop, new_consts

    # final evaluation
    fitnesses = [fit_fn(ind, X, y, c) for ind, c in zip(pop, consts_list)]
    best_i = min(range(len(pop)), key=lambda i: fitnesses[i])
    return pop[best_i], consts_list[best_i], fitnesses[best_i]

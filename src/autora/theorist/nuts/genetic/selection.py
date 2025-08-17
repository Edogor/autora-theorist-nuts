# src/autora/theorist/nuts/genetic/selection.py
import random

def tournament(pop, fitnesses, k: int):
    idxs = random.sample(range(len(pop)), k)
    best = min(idxs, key=lambda i: fitnesses[i])
    return pop[best]

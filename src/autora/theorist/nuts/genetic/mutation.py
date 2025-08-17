# src/autora/theorist/nuts/genetic/mutation.py
import copy, random

def subtree_mutation(t, grow_fn, p_subtree=0.7):
    t = copy.deepcopy(t)
    # TODO: replace random subtree using grow_fn
    return t

def tweak_constants(consts, sigma=0.1):
    return {k: v + random.gauss(0.0, sigma) for k, v in consts.items()}

import itertools
import random

MOVES = ['F', 'R', 'U', 'L', 'D', 'B', 'F\'', 'R\'', 'U\'', 'L\'', 'D\'', 'B\'']

def gen_scrambles(depth):
    scrambles = itertools.product(MOVES, repeat=depth)
    return map(lambda tup: ' '.join(tup), scrambles)

def gen_validation_scrambles(depth, sample_length):
    if len(MOVES) ** depth <= sample_length:
        return gen_scrambles(depth)
    else:
        return [' '.join(random.choices(MOVES, k=depth)) for i in range(sample_length)]


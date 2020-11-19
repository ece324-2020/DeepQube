#!/usr/bin/env python

import itertools
import sys

import numpy as np
import torch

import cubesim

def _attempt_solve(network, scramble, device):
    cube = cubesim.Cube2()
    network.eval()
    with torch.no_grad():
        cube.reset()
        cube.load_scramble(scramble)

        move_count = 0
        while not cube.is_solved():
            if move_count == 20:
                return False

            move = network(cube.get_embedding(device)).argmax()
            cube.moves[move]()
            move_count += 1

        return move_count

def validate(network, scrambles, device):
    solutions = map(lambda scramble: _attempt_solve(network, scramble, device), scrambles)

    solved = 0
    total = 0
    for solution in solutions:
        total += 1
        if solution:
            solved += 1

    return solved / total

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network_file = sys.argv[1]
    scramble_file = sys.argv[2]

    with open(scramble_file) as f:
        scrambles = list(map(lambda s: s.strip(), f.readlines()))

    network = torch.load(network_file, map_location=device)
    accuracy = validate(network, scrambles, device)

    print(accuracy)
    

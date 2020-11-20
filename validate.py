#!/usr/bin/env python

import csv
import argparse
import torch
import cubesim
from baseline import baseline_solver


def _attempt_solve(network, scramble, max_moves, device):
    cube = cubesim.Cube2()
    network.eval()
    with torch.no_grad():
        cube.reset()
        cube.load_scramble(scramble)

        move_count = 0
        while not cube.is_solved():
            if move_count > max_moves:
                return max_moves

            move = network(cube.get_embedding(device)).argmax()
            cube.moves[move]()
            move_count += 1

        return move_count


def validate(network, scrambles, max_moves, device, mode='validation', filename=''):
    def solve(scramble): return _attempt_solve(network, scramble, max_moves, device)
    solutions = map(solve, scrambles)
    lengths = []

    solved = 0
    total = 0
    for solution in solutions:
        lengths.append(solution)
        total += 1
        if solution <= max_moves:
            solved += 1

    if mode == 'comparison':
        baseline_lengths = []
        for scramble in scrambles:
            baseline_lengths.append(baseline_solver(scramble, 'qtm'))

        rows = zip(lengths, baseline_lengths)

        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(('DeepQube', 'Baseline'))
            for row in rows:
                writer.writerow(row)

    return solved / total


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('network', help='Path to network .pt file.')
    parser.add_argument('scramble', help='Path to scramble .txt file.')
    parser.add_argument('moves', help='Maximum number of moves.', type=int)
    parser.add_argument('--filename', help='Specify a filename to enter comparison mode against baseline solver. Outputs results to CSV file.')
    args = parser.parse_args()

    network_file = args.network
    scramble_file = args.scramble
    max_moves = args.moves

    with open(scramble_file) as f:
        scrambles = list(map(lambda s: s.strip(), f.readlines()))

    network = torch.load(network_file, map_location=device)

    accuracy = validate(network, scrambles, device,
                        mode='comparison' if args.filename else 'validation',
                        filename=args.filename or '')
    print(accuracy)

#!/usr/bin/env python

import argparse
import csv

from tqdm import tqdm
import torch

from baseline import baseline_solver
from cubesim.visualizer import print_cube
import cubesim

top_solves = 0
bot_solves = 0

def solve(model1, model2, scramble, max_moves):
    global top_solves
    global bot_solves

    cube = cubesim.Cube2()
    cube.load_scramble(scramble)

    solution = []
    while not cube.layer_solved() and len(solution) < max_moves:
        move = model1(cube.get_embedding()).argmax()
        solution.append(move)
        cube.moves[move]()

    if cube.layer_solved():
        top_solves += 1

    if cube.is_solved():
        return True, solution

    while not cube.is_solved() and len(solution) < max_moves:
        move = model2(cube.get_embedding()).argmax()
        solution.append(move)
        cube.moves[move]()

    if cube.is_solved():
        bot_solves += 1

    if len(solution) < max_moves:
        return True, solution
    else:
        return False, solution

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate the solution to a cube')
    parser.add_argument('--max', type=int, help='max moves', default=40)
    parser.add_argument('model1', type=str, help='path to the reinforcement learning model')
    parser.add_argument('model2', type=str, help='path to the supervised learning model')
    parser.add_argument('scrambles', type=str, help='path to file with scrambles to test')
    parser.add_argument('output', type=str, help='path to output csv')
    
    args = parser.parse_args()
    model1 = torch.load(args.model1, map_location='cpu')
    model2 = torch.load(args.model2, map_location='cpu')

    with open(args.scrambles) as f:
        scrambles = list(map(lambda s: s.strip(), f.readlines()))

    print('evaluating baseline:')
    baseline_lengths = list(map(lambda s: baseline_solver(s, 'qtm'), scrambles))
    print('evaluating model:')
    model_sols = map(lambda s: solve(model1, model2, s, args.max), tqdm(scrambles))
    model_lengths = map(lambda r: len(r[1]) if r[0] else 9999, model_sols)

    rows = zip(model_lengths, baseline_lengths)
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('DeepQube', 'Baseline'))
        writer.writerows(rows)

    print("model1: ", top_solves, "model2: ", bot_solves)


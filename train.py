#!/usr/bin/env python

import itertools

import numpy as np
import torch
import torch.nn.functional as F

import network.rewards
from network.agent import Agent

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO: use argparse
    gamma = 0.99
    epsilon = 0.05
    num_steps = 100
    batch_size = 128
    replay_size = 10000
    reward_fn = lambda x: 0 # TODO: Fix the reward_fn
    nn_params = { 'layers_dim': [4096, 2048, 1024], 'activation': F.relu }

    torch.manual_seed(0)

    with open('./data/4moves.txt', 'r') as f:
        scrambles = list(map(lambda s: s.strip(), f.readlines()))

    print(f'loaded {len(scrambles)} scrambles')
    reward = network.rewards.Naive(device=device)

    agent = Agent(replay_size, reward, device, nn_params)
    model = agent.model
    optimizer = torch.optim.RMSprop(model.parameters(),lr=0.001)
    criterion = torch.nn.SmoothL1Loss()

    scrambles = ['F']
    for i, scramble in enumerate(itertools.cycle(scrambles)):
        losses = agent.play_episode(optimizer, criterion,
                scramble, batch_size, gamma, epsilon, num_steps, device)
        print(i, np.mean(losses))
        torch.save(model, f'checkpoints/{i}.pt')


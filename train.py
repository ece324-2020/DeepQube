#!/usr/bin/env python

import torch
import torch.nn.functional as F

from network.agent import Agent

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO: use argparse
    gamma = 0.99
    num_steps = 1000
    batch_size = 32
    replay_size = 10000
    reward_fn = lambda x: 0 # TODO: Fix the reward_fn
    nn_params = { 'layers_dim': [4096, 2048, 1024], 'activation': F.relu }

    with open('./data/4moves.txt', 'r') as f:
        scrambles = f.readlines()

    print(f'loaded ${len(scrambles)} scrambles')

    agent = Agent(replay_size, reward_fn, device, nn_params)
    model = agent.model
    optimizer = torch.optim.RMSprop(model.parameters())
    criterion = torch.nn.SmoothL1Loss()

    for scramble in scrambles:
        losses = agent.play_episode(optimizer, criterion,
                scramble, batch_size, gamma, num_steps)



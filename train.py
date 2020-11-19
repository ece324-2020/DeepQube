#!/usr/bin/env python

import itertools

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import network.rewards
from network.agent import Agent
from network.exploration import ExplorationRate

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # TODO: use argparse
    gamma = 0.99
    epsilon_scheduler = ExplorationRate(1, 0.1, 10000)
    num_episodes = 5000000
    num_steps = 20
    batch_size = 128
    replay_size = 10000
    target_update_int = 10
    nn_params = { 'layers_dim': [4096, 2048, 1024], 'activation': F.relu }

    save_int = 10000
    torch.manual_seed(0)

    with open('./data/4moves.txt', 'r') as f:
        scrambles = list(map(lambda s: s.strip(), f.readlines()))

    print(f'loaded {len(scrambles)} scrambles')
    reward = network.rewards.Naive(device=device)

    agent = Agent(replay_size, reward, device, nn_params)
    target_net = agent.target_net
    policy_net = agent.policy_net
    optimizer = torch.optim.RMSprop(policy_net.parameters(),lr=0.001)
    criterion = torch.nn.SmoothL1Loss()

    scrambles = ['F', 'R', 'U', 'L', 'D', 'B']
    for i, scramble in enumerate(tqdm(itertools.islice(
            itertools.cycle(scrambles), num_episodes))):
        epsilon = epsilon_scheduler.get_rate(i)
        losses = agent.play_episode(optimizer, criterion,
                scramble, batch_size, gamma, epsilon, num_steps, device)
        print(np.mean(losses))

        if i % target_update_int == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i % save_int == 0:
            torch.save(target_net, f'checkpoints/{i}.pt')



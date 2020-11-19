#!/usr/bin/env python

import argparse
import itertools
import math
import random
import sys

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from data.gen_scrambles import gen_scrambles, gen_validation_scrambles
from network.agent import Agent
from network.exploration import ExplorationRate
import network.rewards
import validate

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='run training loop')
    parser.add_argument('--lr', help='learning rate', default=0.001)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--scheduler', help='epsilon scheduler to change exploration rate',
                        type=tuple, default=(1, 0.1, 10000))
    parser.add_argument('--episodes', help='number of episodes', default=5000000)
    parser.add_argument('--steps', help='maximum number of moves the agent is allowed to make', default=20)
    parser.add_argument('--batch', help='batch size', default=128)
    parser.add_argument('--replay', help='replay size', default=10000)
    parser.add_argument('--update', help='interval at which target network updates', default=10)
    parser.add_argument('--save', help='interval at which .pt checkpoint files are saved', default=10000)
    parser.add_argument('--layers', help='dimensions of the three fully-connected layers',
                        type=tuple, default=(4096, 2048, 1024))

    args = parser.parse_args()
    lr = args.lr
    gamma = args.gamma
    epsilon_scheduler = ExplorationRate(args.scheduler)
    num_episodes = args.episodes
    num_steps = args.steps
    batch_size = args.batch
    replay_size = args.replay
    target_update_int = args.update
    nn_params = {'layers_dim': args.layers, 'activation': F.relu}

    save_int = args.save
    torch.manual_seed(0)
    random.seed(0)

    reward = network.rewards.Naive(device=device)

    agent = Agent(replay_size, reward, device, nn_params)
    target_net = agent.target_net
    policy_net = agent.policy_net
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=lr)
    criterion = torch.nn.SmoothL1Loss()

    pbar = tqdm(total=num_episodes)

    validation_len = 100

    episode = 0
    current_depth = 1
    while True:
        recent_solves = np.zeros(20, dtype=np.bool)
        print(f'training against all {current_depth} move scrambles', file=sys.stderr)
        for scramble in itertools.cycle(gen_scrambles(current_depth)):
            epsilon = epsilon_scheduler.get_rate(episode)

            losses = agent.play_episode(optimizer, criterion,
                    scramble, batch_size, gamma, epsilon, num_steps, device)

            if episode % save_int == 0:
                torch.save(target_net, f'checkpoints/{episode}.pt')

            train_loss = np.mean(losses)
            if episode % target_update_int == 0:
                target_net.load_state_dict(policy_net.state_dict())
                val_acc = validate.validate(target_net,
                        gen_validation_scrambles(current_depth, validation_len),
                        current_depth, device)

                print(f"{episode}\t{train_loss}\t{val_acc}")

                recent_solves = np.roll(recent_solves, -1)
                recent_solves[-1] = math.isclose(val_acc, 1.0)

                if np.sum(recent_solves) > 18:
                    current_depth += 1
                    break

            episode += 1
            pbar.update(1)

        torch.save(target_net, f'checkpoints/{episode}.pt')

    del pbar

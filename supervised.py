#!/usr/bin/env python

import argparse
import functools

from tqdm import trange
import numpy as np
import torch

import cubesim
import network.models

class PathwayDataset(torch.utils.data.Dataset):
    def __init__(self, path, device):
        self.count, states, moves = self._process(path)
        self.states = states.to(device)
        self.moves = moves.to(device)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return self.states[idx, :], self.moves[idx]

    def _process(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            count = len(lines)

        cube = cubesim.Cube2()
        states = torch.zeros(count, 144)
        moves = torch.LongTensor(count)
        for i, line in enumerate(lines):
            state, move = line.split('\t')
            cube.load_state(state)
            move = int(move)
            states[i, :] = cube.get_embedding()
            moves[i] = move

        return count, states, moves

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='finish the network with a supervised training')
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--epochs', help='number of epochs', type=int, default=1000)
    parser.add_argument('--batch', help='batch size', type=int, default=128)
    parser.add_argument('--save', help='interval at which .pt checkpoint files are saved', default=10000)
    parser.add_argument('--layers', help='dimensions of the three fully-connected layers',
                        type=tuple, default=(4096, 2048, 1024))
    parser.add_argument('solutions', type=str, help='solutions data file')
    parser.add_argument('checkpoint', type=str, nargs='?', help='checkpoint file to start with', default=None)


    args = parser.parse_args()
    lr = args.lr
    epochs = args.epochs
    batch = args.batch
    save_int = args.save

    dataset = PathwayDataset(args.solutions, device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch)

    if args.checkpoint == None:
        model = network.models.DQN(144, 12, args.layers).to(device)
    else:
        model = torch.load(args.checkpoint, map_location=device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for i in trange(epochs):
        avg_loss = 0.0
        avg_acc = 0.0
        iters = 0
        for states, moves in dataloader:
            pred = model(states)
            loss = criterion(pred, moves)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += float(loss)
            avg_acc += float((pred.argmax(dim=1) == moves).sum())
            iters += 1
        
        avg_loss /= iters
        avg_acc /= iters
        print(f'{i}\t{avg_loss}\t{avg_acc}')

        if i % save_int:
            torch.save(model, f'supervised_checkpoints/{i}.pt')


import numpy as np
import torch

import cubesim

final_rew = 100.0
neg_rew = -1.0


class Cosine(object):
    def __init__(self, final_rew=100.0, neg_rew=-1.0, device='cpu'):
        cube = cubesim.Cube2()
        self.target = cube.get_embedding(device)

    def __call__(self, embedding):
        current = embedding.squeeze()
        cos_sim = np.dot(current, self.target) / (np.linalg.norm(current)
                                                  * np.linalg.norm(self.target))
        return (cos_sim*final_rew - neg_rew).item()


class Naive(object):
    def __init__(self, final_rew=100.0, neg_rew=-1.0, device='cpu'):
        cube = cubesim.Cube2()
        self.target = cube.get_embedding(device)

    def __call__(self, embedding):
        current = embedding.squeeze()
        if torch.equal(current, self.target):
            return final_rew
        else:
            return neg_rew


class Layer(object):
    def __init__(self, final_rew=100.0, neg_rew=-1.0, device='cpu'):
        cube = cubesim.Cube2()
        embedding = cube.get_embedding(device).reshape(6, 4, 6)
        self.yellow = embedding[5][0]
        self.orange = embedding[0][0]
        self.green = embedding[1][0]
        self.red = embedding[2][0]
        self.blue = embedding[3][0]

    def partial_layer(self, cube):
        reward = 0
        if (torch.equal(cube[5][0], self.yellow)
            and torch.equal(cube[1][2], self.green)
                and torch.equal(cube[0][3], self.orange)):
            reward += 1

        if (torch.equal(cube[5][1], self.yellow)
            and torch.equal(cube[1][3], self.green)
                and torch.equal(cube[2][2], self.red)):
            reward += 1

        if (torch.equal(cube[5][3], self.yellow)
            and torch.equal(cube[2][3], self.red)
                and torch.equal(cube[3][2], self.blue)):
            reward += 1

        if (torch.equal(cube[5][2], self.yellow)
            and torch.equal(cube[0][2], self.orange)
                and torch.equal(cube[3][3], self.blue)):
            reward += 1

        return True if reward >= 2.0 else False

    def __call__(self, embedding):
        current = embedding.squeeze().reshape(6, 4, 6)
        if self.partial_layer(current):
            return final_rew
        else:
            return neg_rew

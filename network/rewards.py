import numpy as np
import torch
import copy
from numpy import linalg as LA

import cubesim

final_rew = 1000.0
neg_rew = -1.0

class Cosine(object):
    def __init__(self, final_rew=1000.0, neg_rew=-1.0, device='cpu'):
        cube = cubesim.Cube2()
        self.target = cube.get_embedding(device)

    def __call__(self, embedding):
        current = embedding.squeeze()
        cos_sim = np.dot(current, self.target) / (np.linalg.norm(current)
                * np.linalg.norm(self.target))
        return (cos_sim*final_rew - neg_rew).item()

class Naive(object):
    def __init__(self, final_rew=1000.0, neg_rew=-1.0, device='cpu'):
        cube = cubesim.Cube2()
        self.target = cube.get_embedding(device)

    def __call__(self, embedding):
        current = embedding.squeeze()
        if torch.equal(current, self.target):
            print('Solved')
            return final_rew
        else:
            return neg_rew

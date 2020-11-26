import numpy as np
import torch
import copy
from numpy import linalg as LA

import cubesim

final_rew = 10.0
neg_rew = -1.0

class Cosine(object):
    def __init__(self, final_rew=10.0, neg_rew=-1.0, device='cpu'):
        cube = cubesim.Cube2()
        self.target = cube.get_embedding(device)

    def __call__(self, embedding,prevembed):
        current = embedding.squeeze()
        if torch.equal(current, self.target):
            return final_rew
        past = prevembed.squeeze()
        cos_sim = np.dot(current, self.target) / (np.linalg.norm(current)
                * np.linalg.norm(self.target))
        prev_sim = np.dot(past, self.target) / (np.linalg.norm(past)
                * np.linalg.norm(self.target))
        if cos_sim > prev_sim:
            return 0.0
        else:
            return neg_rew

class Naive(object):
    def __init__(self, final_rew=10.0, neg_rew=-1.0, device='cpu'):
        cube = cubesim.Cube2()
        self.target = cube.get_embedding(device)

    def __call__(self, embedding,prevembed,terminate):
        
        prev = prevembed.squeeze()
        current = embedding.squeeze()
        if torch.equal(current, self.target):
            return final_rew
        elif terminate == True:
            return (-1*final_rew)
        else:
            return neg_rew

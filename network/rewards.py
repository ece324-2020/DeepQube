import numpy as np
import torch
import copy
from numpy import linalg as LA

def get_reward(cube,rewtype):
	'''
	Input: cube2 class object, type of reward function
	Output: Reward as float

	Naive Reward: return negative reward if not final state else return final reward 

	Cosine Reward: Find cosine simiarility between current state and target state
	Returns cos_sim*final reward - negative reward
	'''

	current = cube.get_embedding().numpy().flatten()

	cubecopy = copy.deepcopy(cube)
	cubecopy.reset()
	target = cubecopy.get_embedding().numpy().flatten()

	#Hyper Parameters
	final_rew = 100.0
	neg_rew = -1.0

	if rewtype == 'Cosine':
		cos_sim = np.dot(current,target) / (np.linalg.norm(current) * np.linalg.norm(target))
		return (cos_sim*final_rew - neg_rew)
	else:
		if current != target:
			return neg_rew
		else:
			return final_rew
	return False


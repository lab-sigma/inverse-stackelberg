import numpy as np
import pickle, os
from numpy.linalg import matrix_rank


def logit(x, V, exp_lambda):
	weights = exp_lambda * (x @ V)
	weights -= np.max(weights)
	exp_rewards = np.exp( weights )
	p_t =  exp_rewards /  np.sum(exp_rewards)
	return p_t


def load_game(filename):
	with open(filename, 'rb') as f:
		game = pickle.load(f)
	return game

class Game:
	def __init__(self, U, V):
		self.leader_actions = V.shape[0]
		self.follower_actions = V.shape[1]
		self.U = U
		self.V = V

	def save(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	def query_response(self, x, exp_lambda, realized=True):
		res = logit(x, self.V, exp_lambda)
		if realized: ### whether or not use realized action
			action, = np.random.choice(self.follower_actions, 1, p=res)
			res = np.eye(self.follower_actions)[action] 
		return res

def generate_random_utility(m, n):
	utility = np.random.normal(loc=0, scale=2, size=(m, n))
	utility -= np.min(utility)
	utility /= np.max(utility)
	return utility

def generate_stackelberg(leader_actions, follower_actions, alpha):
	U = np.eye(leader_actions, follower_actions)
	V = alpha * np.eye(leader_actions, follower_actions) + (1-alpha) * generate_random_utility(leader_actions, follower_actions)
	return Game(U, V)


def generate_lowrank(leader_actions, follower_actions, rank, alpha = 0.2):
	U = np.eye(leader_actions, follower_actions)
	V = alpha * np.eye(rank, follower_actions) + (1-alpha) * generate_random_utility(rank, follower_actions)                
	vecs = []
	for i in range(rank, leader_actions):
		a = np.random.uniform(0,1, size=(rank))
		a /= np.sum(a)
		b =  a @ V
		vecs.append(b)
	vecs = np.stack(vecs)
	V = np.concatenate((V, vecs), axis=0 )
	return Game(U, V)

def generate_securitygame(targets):
	U = np.identity(targets) 
	w = np.random.uniform(-3, -1, size=targets)
	c = np.random.uniform(1, 3, size=targets)
	V = np.diag(w) + c 
	V -= np.min(V)
	V /= np.max(V)
	return Game(U, V)

if __name__ == "__main__":
	game_dir = "game-profile/"
	os.makedirs(game_dir)
	### game of different alpha
	leader_actions = 10
	follower_actions = 10
	for alpha in [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]:
		for n in range(1,6):
			g = generate_stackelberg(leader_actions, follower_actions, alpha)
			filename = f"{game_dir}{leader_actions}x{follower_actions}-{alpha}#{n}"
			g.save(filename)

	### game of different ranks
	leader_actions = 20
	follower_actions = 20
	for rank in [1, 2, 4, 6, 8, 16]:
		for n in range(1,6):
			g = generate_lowrank(leader_actions, follower_actions, rank)
			filename = f"{game_dir}LR{rank}-{leader_actions}x{follower_actions}#{n}"
			g.save(filename)


	### game of different size
	alpha = 0.2
	for leader_actions, follower_actions in [(10, 20), (20, 10),(20, 50),(20, 100), (50, 10),(50, 20),(50, 100), (100, 10),(100, 20),(100, 50)]:
		for n in range(1,6):
			g = generate_stackelberg(leader_actions, follower_actions, alpha)
			filename = f"{game_dir}{leader_actions}x{follower_actions}-{alpha}#{n}"
			g.save(filename)


    ### security game of different number of targets
	for num_targets in [10, 20, 50, 100]: 
		for n in range(1,6):
			g = generate_securitygame(num_targets)
			filename = f"{game_dir}sec-{num_targets}#{n}"
			g.save(filename)

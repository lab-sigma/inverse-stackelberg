import numpy as np
import matplotlib.pyplot as plt
import pickle
import cvxpy as cp

np.set_printoptions(suppress=True)

def logit_dist(V1, V2, cumulative=True):
	m, n = V1.shape
	total = []
	f = 0
	z = cp.Variable(shape=(m))
	for i in range(m):
		f += cp.norm1( V1[i,:] - V2[i,:] - z[i] )
	obj = cp.Minimize(f) 

	lp = cp.Problem(obj)
	lp.solve(solver="MOSEK", verbose=False)
	z = z.value 

	for i in range(m):
		total.append(np.linalg.norm(V1[i,:] - V2[i,:] - z[i], ord=1))

	if cumulative:
		total = sum(total) / (m*n)
	return total


def find_latest(prefix, suffix):
	if not os.path.exists(prefix):
	    os.makedirs(prefix)
	    print("created directory", prefix)
	i = 0
	while os.path.exists(f'{prefix}{i}{suffix}'): 
		i += 1	
	return i
	
class logger:
	def __init__(self, filename, game, para):
		self.p_t_loss = []
		self.payoff_loss = []
		self.p_t_loss_true = []
		self.utility_loss = []
		self.logging_dir = filename
		self.game = game
		self.para = para
		self.checkpoint = []

	def write(self, text):
		print(text)
		with open(self.logging_dir+ '.log', 'a') as f:
			f.write(text)

	def log_round(self, r_est, t):
		self.payoff_loss.append( logit_dist(r_est, self.game.V) )
		# print(self.payoff_loss[-1])
		# print(self.game.V)
		self.checkpoint.append(t)

	def save_results(self, ns, p_t_trues, exp_lambda):
		with open(self.logging_dir+ '.pickle', 'wb') as f:
			pickle.dump((ns, p_t_trues, exp_lambda), f)


	def plot(self):
		fig = plt.figure()

		val = np.asarray(self.payoff_loss)
		plt.plot(self.checkpoint,  val , color='tab:orange', label='payoff distance' )
		for s in self.checkpoint:
			plt.axvline(s, color='green', linestyle='-', alpha=0.2)
		plt.xlim(0)
		plt.ylim(0)
		plt.ylabel("Avg. eps")
		plt.xlabel("Sample Complexity")
		plt.title(f'PAC Learning Curve')
		
		with open(self.logging_dir+ '_hist.pickle', 'wb') as f:
			pickle.dump((self.checkpoint, val), f)

		fig.savefig(f'{self.logging_dir}.png')
		print(f'{self.logging_dir}.png')



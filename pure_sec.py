import numpy as np
import cvxpy as cp

import math, random


def solve_sec_payoffs(xt_hist, pt_hist, exp_lambda):
	f = 0
	num_targets, = xt_hist[0].shape
	w = cp.Variable( shape=(num_targets) )
	c = cp.Variable( shape=(num_targets) )
	V = cp.diag(w) +  cp.vstack([c for i in range(num_targets)])
	for x, p in zip(xt_hist, pt_hist):
		z = cp.multiply(w, x) + c
		z = z * exp_lambda
		### cross entropy
		f += cp.matmul(p, z) - cp.log_sum_exp(z)
	obj = cp.Maximize(f)

	lp = cp.Problem(obj, [V >=0, V <= 1])
	
	lp.solve(solver="MOSEK", verbose=False)

	res = np.round(V.value, 3) ### to avoid -eps in the solutions
	return res


def pure_sec(g, L, exp_lambda, args):
	leader_actions = g.leader_actions
	follower_actions = g.follower_actions

	pt_hist = []
	xt_hist = []
	ids = []
	samples_hist = []
	xt = np.zeros(leader_actions).astype(np.float64)

	for i in range(leader_actions):
		xt[i] += 1
		xt_hist.append( xt.copy() )
		xt[i] -= 1
		ids.append(i)
		samples_hist.append(0)
		pt_hist.append( np.zeros(follower_actions).astype(np.float64) )

	V_est = np.zeros((leader_actions, follower_actions)).astype(np.float64)

	p_t_hat = np.zeros(follower_actions).astype(np.float64)
	for t in range(args.T):
		idx = ids[t%len(ids)]
		xt = xt_hist[idx]

		feedback = g.query_response(xt, exp_lambda, realized=args.realized )

		### use feedback to update empricial distribution
		num_samples = samples_hist[idx]
		p_t_hat =  pt_hist[idx] * (num_samples / (num_samples+1))  
		p_t_hat += feedback / (num_samples+1)
		pt_hist[idx] = p_t_hat
		samples_hist[idx] += 1

		if (t+1) % args.update_period == 0:
			V_est = solve_sec_payoffs(xt_hist, pt_hist, exp_lambda)
			L.log_round(V_est, t)




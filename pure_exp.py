import numpy as np
import cvxpy as cp
import scipy
from scipy.stats import beta
 
def solve_payoffs(xt_hist, pt_hist, samples_hist, exp_lambda, ids):
    f = 0
    leader_actions, = xt_hist[0].shape
    follower_actions, = pt_hist[0].shape
    V = cp.Variable( shape=(leader_actions,follower_actions) )

    total = len(samples_hist)
    threshold = sorted(samples_hist)[-100] if total > 100 else total

    idx = 0
    for x, p, samples in zip(xt_hist, pt_hist, samples_hist):
        if samples < threshold and idx not in ids: continue 
        z = exp_lambda * cp.matmul(x,V) 
        ### cross entropy
        f += cp.matmul(p, z) - cp.log_sum_exp(z)
        idx += 1
    obj = cp.Maximize(f)

    constraints = [V >=0, V <= 1]

    lp = cp.Problem(obj, constraints)
    lp.solve(solver="MOSEK", verbose=False)

    res = np.round(V.value, 3) ### to avoid -eps in the solutions
    return res

def check(pt,  num_samples, t):
    val = np.sort(pt)[-2:]
    rv = beta(num_samples*val[0]+1, num_samples*val[1]+1)
    if rv.cdf(0.5) > 0.9999:
        return np.argmax(pt)
    return -1

def new(n, idx, pt_hist, xt_hist):

    xt = np.random.uniform(size=xt_hist[0].shape[0])
    xt += xt_hist[idx]
    xt /= np.sum(xt)
    return xt


def pure_exp(g, L, exp_lambda, args):
    leader_actions = g.leader_actions
    follower_actions = g.follower_actions

    pt_hist = []
    xt_hist = []
    ids = []
    best_ids = []
    samples_hist = []
    xt = np.zeros(leader_actions).astype(np.float64)
    for i in range(leader_actions):
        xt[i] += 1
        xt_hist.append( xt.copy() )
        xt[i] -= 1
        ids.append(i)
        best_ids.append(i)
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

        if samples_hist[idx] % 100 == 0:
            n = check(pt_hist[idx],  samples_hist[idx], t)
            if samples_hist[idx] > samples_hist[best_ids[t%len(ids)]]:
                best_ids[t%len(ids)] = idx

            if n != -1:
                xt = new(n, best_ids[t%len(ids)], pt_hist, xt_hist)
                print(idx, pt_hist[idx],  samples_hist[idx], xt)

                xt_hist.append(xt)  
                samples_hist.append(0)
                pt_hist.append( np.zeros(follower_actions).astype(np.float64) )
                ids[t%len(ids)] = len(xt_hist)-1

        if (t+1) % args.update_period == 0:
            V_est = solve_payoffs(xt_hist, pt_hist, samples_hist,  exp_lambda, best_ids)
            L.log_round(V_est, t)






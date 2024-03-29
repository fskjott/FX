import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy

# Set random seed for reproducibility
np.random.seed(42)

# Function to simulate correlated Brownian motions
def simulate_correlated_brownian_motion(num_paths, num_steps, correlation_matrix):
    # Generate independent standard normal random variables
    independent_randoms = np.random.normal(size=(num_steps, num_paths))
    
    # Perform Cholesky decomposition on the correlation matrix
    cholesky_matrix = np.linalg.cholesky(correlation_matrix)
    
    # Generate correlated standard normal random variables
    correlated_randoms = np.matmul(cholesky_matrix, independent_randoms.T).T
    
    # Compute the cumulative sum to get Brownian motions
    brownian_motion = np.cumsum(correlated_randoms, axis=0)
    
    return brownian_motion

def evaluate_hedge(start_position, hedge, data):
    # new position
    new_position = start_position + hedge
    # pl var
    start_var = np.matmul(start_position, data.diff()[1:].T).var()
    hedge_var = np.matmul(hedge, data.diff()[1:].T).var()
    new_var = np.matmul(new_position, data.diff()[1:].T).var()
    # hedge costs
    hedge_cost = np.sum(np.abs(hedge))/np.sum(np.abs(start_position))
    hedge_ratio = 1 - new_var / start_var

    return {'cost': hedge_cost, 'ratio': hedge_ratio}

# Parameters
num_paths = 3
num_steps = 100000
correlation_matrix = np.array([[1.0, -0.99, 0.0],
                               [-0.99, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])

sim = simulate_correlated_brownian_motion(num_paths, num_steps, correlation_matrix)

df = pd.DataFrame(sim, columns=["a", "b", "c"])

# Risk vector
start_position = [3, 2, 1]

moves = np.matmul(start_position, df.diff().T)
# old school estimation of cov (XTX)/(n-1)
# np.matmul(df.diff()[1:].T, df.diff()[1:])/(len(df.diff()[1:])-1)

est_corr = df.diff().corr()
est_cov = df.diff().cov()

# This ought to be expected variance of portfolio w:  w cov wT
np.matmul(np.array(start_position).T, np.matmul(est_cov, np.array(start_position)))
moves.var()

# This does not really make sense, as we would need to be able to trade the underlying process
# instead of the asset?
hedge = np.matmul(start_position, est_corr)

# So using out estimates how do we contruct a portfolio?
np.matmul(est_cov, np.matmul(start_position, est_cov))

corr_eig_val, corr_eig_vec = np.linalg.eig(est_corr)

hedge = np.array([-1, 0, 0])

evaluate_hedge(start_position=start_position, hedge=hedge, data=df)

# imagine want to optimize portfolio similiar such that
# minimize wT /sigma W + hedge_cost * scale
# we need to choose a scale of hedge cost, as different volatility/costs produce very different behavior
# eg if hedge cost = 1 but risk of portfolio is almost always lower than 1, then theres no point
# in ever hedging, as cost of hedge is more than the expected move

scale = .1
start_guess = start_position # start by trying to hedge all

def hedger_loss_func(hedge, start_position, covariance, scale):
    position = start_position + hedge
    return np.matmul(np.array(position).T, np.matmul(covariance, np.array(position))) + scale * np.sum(np.abs(hedge))

hedger_loss_func(hedge, start_position, est_cov, scale)


optimizer = scipy.optimize.minimize(hedger_loss_func, x0 = start_guess, args=(start_position, est_cov, scale))
hedge = optimizer['x']
loss = optimizer['fun']

# Voila!
evaluate_hedge(start_position, hedge, df)
# PLOTS

# df.plot()
# plt.title('Simulated Correlated Brownian Motions')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# Analyse the output
# df.diff().corr()
# df.corr()

# correlation of the diff's only really make sense if its norm distributed?
# QQ-plot
# stats.probplot(df["a"].diff(), plot=plt)
# plt.show()
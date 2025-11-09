"""
This code presents the binomial approximation to a Brownian motion with drift (stock price), 
and uses Monte Carlo simulation of price diffusion to estimate an option value.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

# timestep N, time span T, number of simulation N_sim, drift mu, volatility sigma, initial price X_init
def walk_sim(N_sim, N, mu, T, sigma, X_init):
    delt = T/N
    up = sigma * math.sqrt(delt)
    down = -sigma * math.sqrt(delt)
    p = 1./2. * (1. + mu/sigma * math.sqrt(delt))
    X_new = np.ones(N_sim) * X_init
    ptest = np.zeros(N_sim)
    for i in np.arange(N):
        ptest = np.random.uniform(0,1,N_sim)
        ptest = (ptest <= p)
        X_new = X_new + ptest * up + (1 - ptest) * down
    return X_new

# an example with 100000 simulations, drift 0.4, volatility 0.3, intial price of 17
N = 0.3
delta = 0.4/0.3
expect_mean = 17 + delta * N * 1.75
expect_sd = math.sqrt(N * N * 1.75)
res_100000 = walk_sim(100000, 300, 0.4, 1.75, 0.3, 17)
print("number of simulations = 100000")
print("mean:", np.mean(res_100000), "(expected:", expect_mean, ")")
print("sd:", np.std(res_100000), "(expected:", expect_sd, ")")
x_axis = np.linspace(expect_mean - 7 * expect_sd, expect_mean + 7 * expect_sd, 1000)
plt.subplot(3,1,1)
plt.hist(res_100000, density = True)
plt.plot(x_axis, norm.pdf(x_axis, expect_mean, expect_sd))

#tests the variance of different time steps
N = [100, 200, 400, 800, 1600, 3200, 6400]
T = 2
t = []
means = []
variances = []
for i in N:
    dt = T / i
    t.append(dt)
    dZ =np.random.randn(7000, i) * np.sqrt(dt)
    I = np.sum(dZ ** 2, axis = 1)
    means.append(np.mean(I))
    variances.append(np.var(I))
print("change in t", *t)
print("mean", *means)
print("variance", *variances)

plt.subplot(3,1,2)
plt.plot(t, variances)
plt.xlabel(r'$\Delta t$')
plt.ylabel('variance of I')

#monte carlo simulation of option value, r: drift, T: time span, K: strike price, s0: spot price, sigma = volatility
def MC(r, T, K, s0, sig, M, dt):
    N = T / dt
    s = s0
    for i in np.arange(N):
        phi = np.random.normal(0,1,M)
        s = s + s * (r * dt + phi * sig * np.sqrt(dt))
        s = np.maximum(s, 1e-8)
    payoff = np.maximum(K - s, 0)
    val = np.exp(-r * T) * (sum(payoff)/M)
    return val, payoff

# an example with different number of simulations
M = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 96000, 128000, 192000, 256000]
dt = np.zeros(len(M)) 
for i in np.arange(len(M)):
        dt[i] = MC(0.08, 1, 101, 100, 0.2, M[i], 1/250)[0]
plt.subplot(3,1,3)
plt.plot(M, dt)
plt.xlabel("Number of Simulation")
plt.ylabel("Option Value")
plt.show()
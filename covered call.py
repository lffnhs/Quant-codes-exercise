"""
This code examines the covered call strategy using a call option and underlying asset (assuming Brownian motion)
"""

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as p

def blsprice(price, strike, rate, time, volatility):
    sigma_sqrtT = volatility * np.sqrt(time)

    d1 = 1 / sigma_sqrtT * (np.log(price / strike) + (rate + volatility**2 / 2) * time)
    d2 = d1 - sigma_sqrtT

    phi1 = norm.cdf(d1)
    phi2 = norm.cdf(d2)
    disc = np.exp(-rate * time)
    F = price * np.exp((rate) * time)

    call = disc * (F * phi1 - strike * phi2)
    put = disc * (strike * (1 - phi2) + F * (phi1 - 1))
    return call, put

def covered_call(sig, r, T, s0, s_ini, mu, sim, K):
    s = np.ones(sim) * s0
    p = np.ones(sim)
    option = blsprice(s0, K, r, T, sig)[0]

    s = s * np.exp(((mu - sig**2 / 2) * T) + (sig * np.random.normal(0, 1, sim) * np.sqrt(T)))
    p = s + option * np.exp(r * T) - np.maximum(s - K, 0)
    R = np.log(p / s_ini)
    mean = np.mean(R)
    std = np.std(R)
    VAR = np.quantile(R, 0.05)
    cVAR = np.mean(R[R < VAR])
    return R, mean, std, VAR, cVAR

# an example
K_101 = covered_call(0.35, 0.07, 1.5, 100, 100, 0.09, 80000, 101)
p.hist(K_101[0], density = True, bins = 80)
p.xlabel("Probability")
p.ylabel("Density")
p.title("PDF of log return")
p.show()
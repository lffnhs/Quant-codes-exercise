"""
this code prices European/American options using lattice model and Black-Scholes model
"""

import numpy as np
import scipy.stats as stats

#calculate option prices based on Black Scholes model, parameters are spot price, strike price, free risk rate, time span, and volatility
def blsprice(price, strike, rate, time, volatility):
    sigma_sqrtT = volatility * np.sqrt(time)

    d1 = 1 / sigma_sqrtT * (np.log(price / strike) + (rate + volatility**2 / 2) * time)
    d2 = d1 - sigma_sqrtT

    phi1 = stats.norm.cdf(d1)
    phi2 = stats.norm.cdf(d2)
    disc = np.exp(-rate * time)
    F = price * np.exp((rate) * time)

    call = disc * (F * phi1 - strike * phi2)
    put = disc * (strike * (1 - phi2) + F * (phi1 - 1))
    return call, put

# N represents each time step, while type 0 means call and 1 means put
def European(s0, K, r, T, sigma, N, type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r * dt) - d) / (u - d)
    W = np.zeros(N + 1)
    for i in np.arange(0, N + 1):
        W[i] = s0 * (u ** (i)) * (d ** (N - i))
    if type == 0:
        W  = np.maximum(W - K, 0)
    else:
        W = np.maximum(K - W, 0)
    for n in np.arange(N, 0, -1):
        W = np.exp(-r * dt) * (p * W[1:n+1] + (1 - p) * W[0:n])
    return W[0]

def American(s0, K, r, T, sigma, N, type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r * dt) - d) / (u - d)
    W = np.zeros(N + 1)
    S = np.zeros(N + 1)
    for i in np.arange(0, N + 1):
        W[i] = s0 * (u ** (i)) * (d ** (N - i))
        S[i] = s0 * (u ** (i)) * (d ** (N - i))
    if type == 0:
        W  = np.maximum(W - K, 0)
    else:
        W = np.maximum(K - W, 0)
    for n in np.arange(N, 0, -1):
        S = S[:n] / d
        W = np.exp(-r * dt) * (p * W[1:n+1] + (1 - p) * W[0:n])
        if type == 0:
            E = np.maximum(S - K, 0)
        else:
            E = np.maximum(K - S, 0)
        W = np.maximum(W, E)
    return W[0]

#test the lattice model accuracy (in comparison to price calculated by BS model) for different time steps
test = [100, 100, 0.04, 1, 0.32]
print("lattice model for European call")
Europ_C = [European(*test, 500, 0), European(*test, 1000, 0), European(*test, 2000, 0)]
dif_C = np.diff(Europ_C).tolist()
for a, b in zip(Europ_C, dif_C):
    print(a, b)
print("exact:", blsprice(*test)[0])

print("lattice model for European put")
Europ_P = [European(*test, 500, 1), European(*test, 1000, 1), European(*test, 2000, 1)]
dif_P = np.diff(Europ_P).tolist()
for a, b in zip(Europ_P, dif_P):
    print(a, b)
print("exact:", blsprice(*test)[1])

print("lattice model for American put")
Ameri_P = [American(*test, 500, 1), American(*test, 1000, 1), American(*test, 2000, 1)]
dif_P = np.diff(Ameri_P).tolist()
for a, b in zip(Ameri_P, dif_P):
    print(a, b)



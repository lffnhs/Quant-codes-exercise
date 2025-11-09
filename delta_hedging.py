"""
this code applies delta hedging to hedge a position of writeing a put option, assuming the stock price follows a brownian motion.
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

def blsdelta(price, strike, rate, time, volatility):
    d1 = 1 / (volatility * np.sqrt(time)) * (np.log(price / strike) + 
                                             (rate + volatility**2 / 2) * time)
    phi = norm.cdf(d1)

    calldelta = phi
    putdelta = phi -1
    return calldelta, putdelta

def delta_hedge(sig, r, T, K, s0, mu, N):
    dt = T/N
    time =[]
    stock = []
    option = []
    alpha = []
    rf = []
    portfolio = []
    stock_holding = []
    time.append(0)
    stock.append(s0)
    option.append(blsprice(s0, K, r, T, sig)[1])
    alpha.append(blsdelta(s0, K, r, T, sig)[1])
    rf.append(blsprice(s0, K, r, T, sig)[1] - blsdelta(s0, K, r, T, sig)[1] * s0)
    portfolio.append(option[0] + alpha[0] * s0 + rf[0])
    stock_holding.append(alpha[0] * stock[0])

    for i in np.arange(1, N):
        time.append(time[i - 1] + dt)
        stock.append(stock[i - 1] * np.exp((mu - sig**2/2) * dt + sig * np.random.normal(0,1)
                                            * np.sqrt(dt)))
        option.append(blsprice(stock[i], K, r, T - time[i], sig)[1])
        alpha.append(blsdelta(stock[i], K, r, T - time[i], sig)[1])
        rf.append(rf[i - 1] * np.exp(r * dt) - (alpha[i] - alpha[i - 1]) * stock[i])
        portfolio.append(-option[i] + alpha[i] * stock[i] + rf[i])
        stock_holding.append(alpha[i] * stock[i])
    E_hedge = np.exp(-r * T) * portfolio[N - 1] / abs(option[0])
    return time, stock, rf, stock_holding, portfolio, E_hedge

# an example
E = []
info = delta_hedge(0.16, 0.06, 2, 100, 100, 0.08, 1000)
print("Relative hedging error:", abs(info[5]))
p.plot(info[0], info[1], info[0], info[2], info[0], info[3], info[0], info[4])
p.legend(labels = ["Stock price", "Risk free account", "Stock holding", "Portfolio value"])
p.title("Number of Hedging Rebalances = 1000")
p.show()


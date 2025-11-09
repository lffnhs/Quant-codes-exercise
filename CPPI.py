"""
this code applies Constant Proportion Portfolio Insurance (CPPI) to balance 
a portfolio's growth potential with downside protection, assuming the market return
follows Browniian motion.
"""

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Inputs: T: time span, sig: volatility, mu: drift, P0: initial portfolio value, 
# r: risk-free rate, B0: initial amount in the risk-free asset, alpha0: initial risky asset allocation
# S0: initial risky asset price, F: floor value, M: multiplier.
# outputs: R: return, mean and std of return, Value at risk, conditional value at risk
def CPPI(T, sig, mu, P0, r, dt, B0, alpha0, S0, sim, F,M):
    N = T / dt
    S = np.zeros(sim) + S0
    alpha = M * (np.maximum(0, B0 * np.exp(r * dt) + alpha0 * S - F) / S)
    B = B0 - alpha * S
    for i in np.arange(N):
        S = S * np.exp(((mu - sig**2 / 2) * dt) + (sig * np.random.normal(0, 1, sim)
                                                   * np.sqrt(dt)))
        alpha_1 = M * (np.maximum(0, B * np.exp(r * dt) + alpha * S - F) / S)
        B = B * np.exp(r * dt) - (alpha_1 - alpha) * S
        alpha = alpha_1
    P = B + alpha * S
    R = np.log(P / P0)
    mean = np.mean(R)
    std = np.std(R)
    VAR = np.quantile(R, 0.05)
    cVAR = np.mean(R[R < VAR])
    return R, mean, std, VAR, cVAR

# examples with different multipliers and floor values
sim_1 = CPPI(2, 0.32, 0.12, 100, 0.06, 1/250, 100, 0, 100, 80000, 0, 0.5)
sim_2 = CPPI(2, 0.32, 0.12, 100, 0.06, 1/250, 100, 0, 100, 80000, 0, 1)
sim_3 = CPPI(2, 0.32, 0.12, 100, 0.06, 1/250, 100, 0, 100, 80000, 0, 2)
sim_4 = CPPI(2, 0.32, 0.12, 100, 0.06, 1/250, 100, 0, 100, 80000, 85, 2)
sim_5 = CPPI(2, 0.32, 0.12, 100, 0.06, 1/250, 100, 0, 100, 80000, 85, 4)

print("F \ t M \t Mean \t\t\t Std \t 95%VAR \t 95%cVAR")
print("0 \t 0.5 \t", sim_1[1], "\t", sim_1[2], "\t", sim_1[3], "\t", sim_1[4])
print("0 \t 1 \t", sim_2[1], "\t", sim_2[2], "\t", sim_2[3], "\t", sim_2[4])
print("0 \t 2 \t", sim_3[1], "\t", sim_3[2], "\t", sim_3[3], "\t", sim_3[4])
print("85 \t 2 \t", sim_4[1], "\t", sim_4[2], "\t", sim_4[3], "\t", sim_4[4])
print("85 \t 4 \t", sim_5[1], "\t", sim_5[2], "\t", sim_5[3], "\t", sim_5[4])

plt.subplot(5,1,1)
plt.hist(sim_1[0], density = True, bins = 200)
plt.xlabel("R")
plt.ylabel("Density")
plt.title("F = 0, M = 0.5")

plt.subplot(5,1,2)
plt.hist(sim_2[0], density = True, bins = 200)
plt.xlabel("R")
plt.ylabel("Density")
plt.title("F = 0, M = 1")

plt.subplot(5,1,3)
plt.hist(sim_3[0], density = True, bins = 200)
plt.xlabel("R")
plt.ylabel("Density")
plt.title("F = 0, M = 2")

plt.subplot(5,1,4)
plt.hist(sim_4[0], density = True, bins = 200)
plt.xlabel("R")
plt.ylabel("Density")
plt.title("F = 85, M = 2")

plt.tight_layout()
plt.show()



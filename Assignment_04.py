#NAME - AYUSH KUMAR AHIRWAR
#ROLL N0. - 240240
#ASSIGNMENT - 04


#QUESTION - 01 
#CODE 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, beta as beta_dist, lognorm

#  DATA
data = pd.read_csv("word-recognition.csv")
rt = data["rt"].values
freq = data["frequency"].values

sigma = 0.4
alpha = 6

# COMMON FUNCTIONS
def log_likelihood(beta_, gamma_, theta_):
    mu = alpha + beta_ * freq
    part1 = theta_ * lognorm.pdf(rt, s=sigma, scale=np.exp(mu))
    part2 = (1 - theta_) * lognorm.pdf(rt, s=sigma, scale=np.exp(gamma_))
    return np.sum(np.log(part1 + part2 + 1e-12))

def log_prior(beta_, gamma_, theta_):
    return (
        np.log(norm.pdf(beta_, 0, 0.5) + 1e-12)
        + np.log(norm.pdf(gamma_, 5, 0.5) + 1e-12)
        + np.log(beta_dist.pdf(theta_, 70, 30) + 1e-12)
    )

# 1.1: IMPORTANCE SAMPLING

N = 2000

beta_prop = np.random.normal(0, 1, N)
gamma_prop = np.random.normal(5, 1, N)
theta_prop = np.random.uniform(0, 1, N)

log_weights = []

for i in range(N):
    lw = log_likelihood(beta_prop[i], gamma_prop[i], theta_prop[i]) + \
         log_prior(beta_prop[i], gamma_prop[i], theta_prop[i])
    log_weights.append(lw)

log_weights = np.array(log_weights)
log_weights -= np.max(log_weights)

weights = np.exp(log_weights)
weights /= np.sum(weights)

idx = np.random.choice(np.arange(N), size=N//2, p=weights)

beta_is = beta_prop[idx]
gamma_is = gamma_prop[idx]
theta_is = theta_prop[idx]

print("\nQ1 Done (Importance Sampling)")

# 1.2: MCMC (Metropolis)

steps = 4000

beta_chain = [0]
gamma_chain = [5]
theta_chain = [0.7]

for i in range(steps):

    beta_new = np.random.normal(beta_chain[-1], 0.1)
    gamma_new = np.random.normal(gamma_chain[-1], 0.1)
    theta_new = np.random.normal(theta_chain[-1], 0.03)

    if theta_new <= 0 or theta_new >= 1:
        theta_new = theta_chain[-1]

    curr = log_likelihood(beta_chain[-1], gamma_chain[-1], theta_chain[-1]) + \
           log_prior(beta_chain[-1], gamma_chain[-1], theta_chain[-1])

    new = log_likelihood(beta_new, gamma_new, theta_new) + \
          log_prior(beta_new, gamma_new, theta_new)

    ratio = np.exp(new - curr)

    if np.random.rand() < ratio:
        beta_chain.append(beta_new)
        gamma_chain.append(gamma_new)
        theta_chain.append(theta_new)
    else:
        beta_chain.append(beta_chain[-1])
        gamma_chain.append(gamma_chain[-1])
        theta_chain.append(theta_chain[-1])

beta_mcmc = np.array(beta_chain[1000:])
gamma_mcmc = np.array(gamma_chain[1000:])
theta_mcmc = np.array(theta_chain[1000:])

print("Q2 Done (MCMC)")

# 1.3: GRAPH COMPARISON

plt.hist(beta_is, bins=30, alpha=0.5, label="IS")
plt.hist(beta_mcmc, bins=30, alpha=0.5, label="MCMC")
plt.legend()
plt.title("Posterior of beta")
plt.show()

print("Q3 Done (Graph plotted)")

# 1.4: HYPOTHESIS CHECK

beta_mean = np.mean(beta_mcmc)

print("\nQ4 Result:")
print("Mean beta:", beta_mean)

if beta_mean > 0:
    print("Hypothesis Supported (beta > 0)")
else:
    print("Hypothesis NOT supported")

# 1.5: THETA INTERPRETATION

theta_mean = np.mean(theta_mcmc)

print("\nQ5 Result:")
print("Mean theta:", theta_mean)
print("Theta close to 1 => attentive, else guessing")


#QUESTION - 02 
#CODE 
import numpy as np
import matplotlib.pyplot as plt


# data
np.random.seed(0)

true_mu = 800
true_var = 100
y = np.random.normal(true_mu, np.sqrt(true_var), 500)

plt.hist(y, bins=30)
plt.title("Histogram of data")
plt.show()


# Gradient function
def gradient(mu, sigma, y, n, m, s, a, b):
    grad_mu = ((n*mu - np.sum(y)) / (sigma**2)) + ((mu - m)/(s**2))
    grad_sigma = (n/sigma) - (np.sum((y-mu)**2)/(sigma**3)) + ((sigma - a)/(b**2))
    return np.array([grad_mu, grad_sigma])


# Potential energy
def V(mu, sigma, y, n, m, s, a, b):
    log_lik = -np.sum(-0.5*np.log(2*np.pi*sigma**2) - ((y-mu)**2)/(2*sigma**2))
    log_prior_mu = -0.5*np.log(2*np.pi*s**2) - ((mu-m)**2)/(2*s**2)
    log_prior_sigma = -0.5*np.log(2*np.pi*b**2) - ((sigma-a)**2)/(2*b**2)
    return log_lik - log_prior_mu - log_prior_sigma


# HMC sampler
def HMC(y, n, m, s, a, b, step, L, initial_q, nsamp, nburn):

    mu_chain = np.zeros(nsamp)
    sigma_chain = np.zeros(nsamp)

    mu_chain[0] = initial_q[0]
    sigma_chain[0] = initial_q[1]

    i = 0
    reject = 0

    while i < nsamp-1:

        q = np.array([mu_chain[i], sigma_chain[i]])
        p = np.random.normal(0, 1, 2)

        current_V = V(q[0], q[1], y, n, m, s, a, b)
        current_T = np.sum(p**2)/2

        # leapfrog
        for j in range(L):
            p = p - (step/2)*gradient(q[0], q[1], y, n, m, s, a, b)
            q = q + step*p
            p = p - (step/2)*gradient(q[0], q[1], y, n, m, s, a, b)

        proposed_V = V(q[0], q[1], y, n, m, s, a, b)
        proposed_T = np.sum(p**2)/2

        # -------- FIXED ACCEPTANCE --------
        delta = current_V + current_T - proposed_V - proposed_T

        if delta > 0:
            accept_prob = 1
        else:
            accept_prob = np.exp(delta)

        # accept/reject
        if np.random.rand() < accept_prob:
            mu_chain[i+1] = q[0]
            sigma_chain[i+1] = abs(q[1])   # sigma negative na ho
        else:
            mu_chain[i+1] = mu_chain[i]
            sigma_chain[i+1] = sigma_chain[i]
            reject += 1

        i += 1

    return mu_chain[nburn:], sigma_chain[nburn:], reject


# Exercise 2.1

mu_post, sigma_post, rej = HMC(
    y=y,
    n=len(y),
    m=1000, s=20,
    a=10, b=2,
    step=0.01,
    L=12,
    initial_q=[1000, 11],
    nsamp=6000,
    nburn=2000
)

print("mu mean:", np.mean(mu_post))
print("sigma mean:", np.mean(sigma_post))

plt.hist(mu_post, bins=30)
plt.title("Posterior of mu")
plt.show()

plt.hist(sigma_post, bins=30)
plt.title("Posterior of sigma")
plt.show()


# Exercise 2.2

for ns in [100, 1000, 6000]:
    nb = ns//3
    mu_p, sigma_p, _ = HMC(y, len(y), 1000, 20, 10, 2, 0.01, 12, [1000,11], ns, nb)

    print("nsamp =", ns, "mu mean =", np.mean(mu_p))

    plt.hist(mu_p, bins=20)
    plt.title(f"mu posterior nsamp={ns}")
    plt.show()


# Exercise 2.3

for st in [0.001, 0.005, 0.01]:
    mu_p, sigma_p, r = HMC(y, len(y), 1000, 20, 10, 2, st, 12, [1000,11], 6000, 2000)

    print("step =", st, "reject =", r)

    plt.hist(mu_p, bins=20)
    plt.title(f"mu posterior step={st}")
    plt.show()


# Exercise 2.4

plt.plot(mu_post[:200])
plt.title("mu chain")
plt.show()

plt.plot(sigma_post[:200])
plt.title("sigma chain")
plt.show()


# Exercise 2.5

priors = [(400,5), (400,20), (1000,5), (1000,20), (1000,100)]

for m_val, s_val in priors:
    mu_p, sigma_p, _ = HMC(y, len(y), m_val, s_val, 10, 2, 0.01, 12, [1000,11], 6000, 2000)

    print("m =", m_val, "s =", s_val, "mu mean =", np.mean(mu_p))

    plt.hist(mu_p, bins=20)
    plt.title(f"mu posterior m={m_val}, s={s_val}")
    plt.show()
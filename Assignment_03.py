#NAME - AYUSH KUMAR AHIRWAR
#ROLL NO. - 240240
#ASSIGNMENT - 3


#Question - 1 
#CODE : 
import numpy as np
import matplotlib.pyplot as plt
from math import comb

print(" Question 1\n")

# -----------------------------
# GIVEN VALUES
# -----------------------------
n = 10
y = 7
marginal = 227/1408   # given in question

# -----------------------------
# FUNCTIONS
# -----------------------------
def likelihood(theta):
    return comb(n,y)*(theta**y)*((1-theta)**(n-y))

def prior(theta):
    if 0.5 <= theta <= 1:
        return 2
    else:
        return 0

def posterior(theta):
    return likelihood(theta)*prior(theta)/marginal


# 1) Posterior values

print("\nPOSTERIOR VALUES\n")

vals = [0.75,0.25,1]

for t in vals:
    print("p(",t,"|y) =",posterior(t))


# 3)  maximum posterior density
theta_grid = np.linspace(0,1,5000)
post_values = [posterior(t) for t in theta_grid]

map_theta = theta_grid[np.argmax(post_values)]

print("\nMAP ESTIMATE =",map_theta)


# 2 & 4) GRAPHS

theta = np.linspace(0,1,400)

like = [likelihood(t) for t in theta]
pri = [prior(t) for t in theta]
post = [posterior(t) for t in theta]

# Posterior only
plt.figure()
plt.plot(theta,post)
plt.title("Posterior Distribution")
plt.xlabel("theta")
plt.ylabel("p(theta|y)")
plt.grid()
plt.show()

# Comparison graph
plt.figure()
plt.plot(theta,like,label="Likelihood")
plt.plot(theta,pri,label="Prior")
plt.plot(theta,post,label="Posterior",linewidth=3)
plt.legend()
plt.title("Likelihood vs Prior vs Posterior")
plt.xlabel("theta")
plt.ylabel("Density")
plt.grid()
plt.show()

#Outcome will come 
#POSTERIOR VALUES
#p( 0.75 |y) = 3.10482344438326 = 3.10
#p( 0.25 |y) = 0.0
#p( 1 |y) = 0.0
#MAP ESTIMATE = 0.6999399879975995 = 0.7

#Question - 2 
#CODE : 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, truncnorm
from scipy.integrate import trapezoid

print ("\n ")
print ("\n Question 2\n")


# =========================
# LOAD DATA
# =========================
data = pd.read_csv("recognition.csv.csv")

Tw = data["Tw"].values
Tnw = data["Tnw"].values

sigma = 60

print("Mean Words =", np.mean(Tw))
print("Mean NonWords =", np.mean(Tnw))
print("Difference =", np.mean(Tnw) - np.mean(Tw))


# 1 â€” Posterior of mu (Null Model)

mu_vals = np.linspace(150, 500, 800)
log_post = []

for mu in mu_vals:
    log_like_w = np.sum(norm.logpdf(Tw, mu, sigma))
    log_like_nw = np.sum(norm.logpdf(Tnw, mu, sigma))
    log_prior = norm.logpdf(mu, 300, 50)

    log_post.append(log_like_w + log_like_nw + log_prior)

log_post = np.array(log_post)
# convert log â†’ normal safely
log_post -= np.max(log_post)
posterior_mu = np.exp(log_post)

posterior_mu /= trapezoid(posterior_mu, mu_vals)

plt.figure()
plt.plot(mu_vals, posterior_mu)
plt.title("Posterior of mu (Null Model)")
plt.xlabel("mu")
plt.ylabel("density")
plt.show()


# 2 â€” Prior Prediction (Lexical Access Model)

N = 5000

mu_samples = np.random.normal(300, 50, N)

a, b = 0/50, np.inf
delta_samples = truncnorm.rvs(a, b, loc=0, scale=50, size=N)

Tw_sim = np.random.normal(mu_samples, sigma)
Tnw_sim = np.random.normal(mu_samples + delta_samples, sigma)

plt.figure()
plt.hist(Tw_sim, bins=40, alpha=0.5, label="Words")
plt.hist(Tnw_sim, bins=40, alpha=0.5, label="Non-words")
plt.legend()
plt.title("Prior Prediction â€” Lexical Model")
plt.show()

# 3 â€” Prior Prediction (Null Model)

mu_samples = np.random.normal(300, 50, N)

Tw_null = np.random.normal(mu_samples, sigma)
Tnw_null = np.random.normal(mu_samples, sigma)

plt.figure()
plt.hist(Tw_null, bins=40, alpha=0.5, label="Words")
plt.hist(Tnw_null, bins=40, alpha=0.5, label="Non-words")
plt.legend()
plt.title("Prior Prediction â€” Null Model")
plt.show()


# 4 â€” Observed Data

plt.figure()
plt.hist(Tw, bins=20, alpha=0.5, label="Real Words")
plt.hist(Tnw, bins=20, alpha=0.5, label="Real Non-words")
plt.legend()
plt.title("Observed Data")
plt.show()

# 5 â€” Posterior of delta (Lexical Model)

from scipy.special import logsumexp

delta_vals = np.linspace(0, 200, 200)
mu_grid = np.linspace(150, 500, 120)

posterior_delta = []

for delta in delta_vals:

    log_vals = []

    for mu in mu_grid:
        log_like_w = np.sum(norm.logpdf(Tw, mu, sigma))
        log_like_nw = np.sum(norm.logpdf(Tnw, mu + delta, sigma))
        log_prior_mu = norm.logpdf(mu, 300, 50)

        # truncated prior: delta > 0
        if delta <= 0:
            continue
        log_prior_delta = norm.logpdf(delta, 0, 50)

        log_vals.append(log_like_w + log_like_nw + log_prior_mu + log_prior_delta)

    if len(log_vals) == 0:
        posterior_delta.append(-np.inf)
    else:
        posterior_delta.append(logsumexp(log_vals))

posterior_delta = np.array(posterior_delta)

# convert safely back
posterior_delta -= np.max(posterior_delta)
posterior_delta = np.exp(posterior_delta)

posterior_delta /= trapezoid(posterior_delta, delta_vals)

plt.figure()
plt.plot(delta_vals, posterior_delta)
plt.title("Posterior of delta (Lexical Model)")
plt.xlabel("delta")
plt.ylabel("density")
plt.show()

"""Outcome will come
Mean Words = 321.3745739134762
Mean NonWords = 323.2388094246616
Difference = 1.8642355111854272"""

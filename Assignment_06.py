#NAME - AYUSH KUMAR AHIRWAR
#ROLL NO. - 240240
#ASSIGNMENT - 06

#PART- 01 
#CODE
print("\n PART-01")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom

# Data
y = np.array([10, 15, 15, 14, 14, 13, 11, 12, 16])
n = 20
sum_y = np.sum(y)
N = len(y)

# Priors
# Model 1
a1_prior, b1_prior = 6, 6
a1 = a1_prior + sum_y
b1 = b1_prior + N*n - sum_y

# Model 2
a2_prior, b2_prior = 20, 60
a2 = a2_prior + sum_y
b2 = b2_prior + N*n - sum_y

# EX 1.1 Posterior Plot
theta = np.linspace(0,1,500)
plt.plot(theta, beta.pdf(theta, a1, b1), label="Model 1")
plt.plot(theta, beta.pdf(theta, a2, b2), label="Model 2")
plt.legend()
plt.title("Posterior Distributions")
plt.show()

# EX 1.2 LPPD
def compute_lppd(a, b):
    theta_samples = np.random.beta(a, b, 1000)
    lppd = 0
    for yi in y:
        probs = binom.pmf(yi, n, theta_samples)
        lppd += np.log(np.mean(probs))
    return lppd

lppd_m1 = compute_lppd(a1, b1)
lppd_m2 = compute_lppd(a2, b2)

print("\nLPPD M1:", lppd_m1)
print("LPPD M2:", lppd_m2)

# EX 1.3 Deviance
dev_m1 = -2 * lppd_m1
dev_m2 = -2 * lppd_m2

print("\nDeviance M1:", dev_m1)
print("Deviance M2:", dev_m2)

# EX 1.4 Best Model
if dev_m1 < dev_m2:
    print("ðŸ‘‰ Model 1 better (in-sample)")
else:
    print("ðŸ‘‰ Model 2 better (in-sample)")

# EX 1.5 New Data
new_data = np.array([5,6,10,8,9])

def compute_lppd_new(a, b, data):
    theta_samples = np.random.beta(a, b, 1000)
    lppd = 0
    for yi in data:
        probs = binom.pmf(yi, n, theta_samples)
        lppd += np.log(np.mean(probs))
    return lppd

lppd_new_m1 = compute_lppd_new(a1, b1, new_data)
lppd_new_m2 = compute_lppd_new(a2, b2, new_data)

dev_new_m1 = -2 * lppd_new_m1
dev_new_m2 = -2 * lppd_new_m2

print("\nNew Data Deviance M1:", dev_new_m1)
print("New Data Deviance M2:", dev_new_m2)

# EX 1.6 LOO-CV
def loo_lppd(a_prior, b_prior):
    total = 0
    for i in range(len(y)):
        y_train = np.delete(y, i)
        y_test = y[i]
        
        a_post = a_prior + np.sum(y_train)
        b_post = b_prior + len(y_train)*n - np.sum(y_train)
        
        theta_samples = np.random.beta(a_post, b_post, 1000)
        probs = binom.pmf(y_test, n, theta_samples)
        
        total += np.log(np.mean(probs))
    return total

loo_m1 = loo_lppd(a1_prior, b1_prior)
loo_m2 = loo_lppd(a2_prior, b2_prior)

print("\nLOO M1:", loo_m1)
print("LOO M2:", loo_m2)

if loo_m1 > loo_m2:
    print("ðŸ‘‰ Model 1 better (LOO)")
else:
    print("ðŸ‘‰ Model 2 better (LOO)")

#PART - 2 
#CODE
print("\n PART-02")
import numpy as np
import math
from scipy.stats import binom

# EX 2.1 Exact ML
def ML_binomial(k, n, a, b):
    return (math.comb(n, k) *
            math.gamma(k+a) * math.gamma(n-k+b) /
            math.gamma(n+a+b) *
            math.gamma(a+b) /
            (math.gamma(a)*math.gamma(b)))

k, n = 2, 10

priors = [(0.1,0.4),(1,1),(2,6),(6,2),(20,60),(60,20)]

print("\n Exact Marginal Likelihood ")
for a,b in priors:
    print(f"Beta({a},{b}):", ML_binomial(k,n,a,b))

# EX 2.2 Monte Carlo ML
def mc_marginal_likelihood(k, n, a, b):
    theta = np.random.beta(a, b, 10000)
    likelihood = binom.pmf(k, n, theta)
    return np.mean(likelihood)

print("\n Monte Carlo Marginal Likelihood ")
for a,b in priors:
    print(f"MC Beta({a},{b}):", mc_marginal_likelihood(k,n,a,b))

    """OUTPUTS
 PART-01

LPPD M1: -18.639715432585824
LPPD M2: -24.37358132819643

Deviance M1: 37.27943086517165
Deviance M2: 48.74716265639286
ðŸ‘‰ Model 1 better (in-sample)

New Data Deviance M1: 49.36121477488086
New Data Deviance M2: 30.274840958963768

LOO M1: -19.465368060345764
LOO M2: -25.739567715670454
ðŸ‘‰ Model 1 better (LOO)

 PART-02

 Exact Marginal Likelihood 
Beta(0.1,0.4): 0.03980886944647779
Beta(1,1): 0.09090909090909091
Beta(2,6): 0.19852941176470587
Beta(6,2): 0.009718222953517071
Beta(20,60): 0.26935910002194535
Beta(60,20): 0.0007989621836746952

 Monte Carlo Marginal Likelihood 
MC Beta(0.1,0.4): 0.040408686016812655
MC Beta(1,1): 0.08949053018550691
MC Beta(2,6): 0.1996523613620166
MC Beta(6,2): 0.009592970118923155
MC Beta(20,60): 0.2693393847657159
MC Beta(60,20): 0.0008080953792249837"""
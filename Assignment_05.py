#NAME - AYUSH KUMAR AHIRWAR
#ROLL NO. - 240240
#ASSIGNMENT - 05

#PART- 01 
#CODE 

# PART 1

import pandas as pd
import numpy as np

#  data
df = pd.read_csv("df_powerpose_f05c6034-0f40-45df-9f3d-2507237af76c.csv")

# Create change variable
df["test_change"] = df["testm2"] - df["testm1"]

# Encode treatment: High = 1, Low = 0
df["hptreat_bin"] = df["hptreat"].map({"High":1, "Low":0})

# Separate groups
high_group = df[df["hptreat_bin"] == 1]["test_change"]
low_group = df[df["hptreat_bin"] == 0]["test_change"]

# Means
mean_high = np.mean(high_group)
mean_low = np.mean(low_group)

# Difference (this is effect)
effect = mean_high - mean_low

print(" PART 1 RESULT ")
print("Mean (High pose):", mean_high)
print("Mean (Low pose):", mean_low)
print("Effect (High - Low):", effect)


#PART - 02 
#CODE 

# EXERCISE 2.1
import numpy as np

def generate_crossings(L, alpha, beta):
    lam = np.exp(alpha + beta * L)
    return np.random.poisson(lam)

# Example
print("Example crossings:", generate_crossings(10, 0.15, 0.25))

# EXERCISE 2.2
n = 1000

alpha_samples = np.random.normal(0.15, 0.1, n)
beta_samples = np.random.normal(0.25, 0.05, n)

L = 4

lam = np.exp(alpha_samples + beta_samples * L)
crossings = np.random.poisson(lam)

print("\nMean crossings (L=4):", np.mean(crossings))

# EXERCISE 2.3
import pandas as pd
import numpy as np

#data 
df2 = pd.read_csv(r"C:\Users\ayush\OneDrive\Documents\crossings_3228b101-2b58-437f-a5ca-4f6325f5f0ee.csv")

# Preprocessing
df2["s.length"] = df2["s.length"] - df2["s.length"].mean()
df2["lang"] = (df2["Language"] == "German").astype(int)
df2["interaction"] = df2["s.length"] * df2["lang"]

# log transform
df2["nCross_log"] = np.log(df2["nCross"] + 1)

# M1 
X1 = np.column_stack((np.ones(len(df2)), df2["s.length"]))
y1 = df2["nCross_log"]

beta_m1 = np.linalg.inv(X1.T @ X1) @ (X1.T @ y1)

# M2 
X2 = np.column_stack((np.ones(len(df2)), df2["s.length"], df2["lang"], df2["interaction"]))
y2 = df2["nCross_log"]

beta_m2 = np.linalg.inv(X2.T @ X2) @ (X2.T @ y2)

print("\nM1 Coefficients:", beta_m1)
print("M2 Coefficients:", beta_m2)

# EXERCISE 2.4 
import numpy as np
import pandas as pd

# Shuffle data
df2 = df2.sample(frac=1, random_state=42).reset_index(drop=True)

folds = [df2.iloc[i::5] for i in range(5)]

error_m1 = 0
error_m2 = 0

for i in range(5):
    
    test = folds[i]
    train = pd.concat([folds[j] for j in range(5) if j != i])
    
    # M1 
    X1_train = np.column_stack((np.ones(len(train)), train["s.length"]))
    y1_train = train["nCross_log"]
    
    beta1 = np.linalg.inv(X1_train.T @ X1_train) @ (X1_train.T @ y1_train)
    
    X1_test = np.column_stack((np.ones(len(test)), test["s.length"]))
    pred1 = X1_test @ beta1
    
    error_m1 += np.sum((test["nCross_log"] - pred1)**2)
    
    
    #M2
    X2_train = np.column_stack((
        np.ones(len(train)),
        train["s.length"],
        train["lang"],
        train["interaction"]
    ))
    
    y2_train = train["nCross_log"]
    
    beta2 = np.linalg.inv(X2_train.T @ X2_train) @ (X2_train.T @ y2_train)
    
    X2_test = np.column_stack((
        np.ones(len(test)),
        test["s.length"],
        test["lang"],
        test["interaction"]
    ))
    
    pred2 = X2_test @ beta2
    
    error_m2 += np.sum((test["nCross_log"] - pred2)**2)

print("\n FINAL RESULT ")
print("Error M1:", error_m1)
print("Error M2:", error_m2)

if error_m2 < error_m1:
    print("ðŸ‘‰ M2 better (Language matters)")
else:
    print("ðŸ‘‰ M1 better (Only length matters)")
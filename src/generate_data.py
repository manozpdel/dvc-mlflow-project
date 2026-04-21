import pandas as pd
import numpy as np
import os

os.makedirs("data/raw", exist_ok=True)

np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + 3 + np.random.randn(100)

df = pd.DataFrame({"X": X.squeeze(), "y": y})
df.to_csv("data/raw/data.csv", index=False)

print("Raw data generated")
import pandas as pd
import os

os.makedirs("data/features", exist_ok=True)

df = pd.read_csv("data/processed/data.csv")

df["X_squared"] = df["X"] ** 2

df.to_csv("data/features/data.csv", index=False)

print("Feature engineered data saved")
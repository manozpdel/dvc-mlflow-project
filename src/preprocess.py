import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/data.csv")

df = df.dropna()
df["X"] = df["X"] / df["X"].max()

df.to_csv("data/processed/data.csv", index=False)

print("Processed data saved")
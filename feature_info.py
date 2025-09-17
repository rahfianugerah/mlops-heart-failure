import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent
train_p = repo_root / "preprocessing" / "preprocess_dataset" / "train_preprocessed.csv"

df = pd.read_csv(train_p)
features = df.columns[:-1]
target = df.columns[-1]

print("Features (order-sensitive):")
for i, f in enumerate(features, 1):
    print(f"{i:2d}. {f}")
print(f"\nTarget: {target}")
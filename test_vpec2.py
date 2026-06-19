import pandas as pd
df = pd.read_csv("results/outputs/stratified_h0.csv")
print("vpecerr describe:")
print(df["vpecerr"].describe())
print("vpec describe:")
print(df["vpec"].describe())
print("z_hd describe:")
print(df["z_hd"].describe())

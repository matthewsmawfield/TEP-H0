import pandas as pd
df = pd.read_csv("data/processed/hosts_processed.csv")
print("z_cmb:", df['z_cmb'].head())
print("vpec:", df['vpec'].head())
print("vpecerr:", df['vpecerr'].head())

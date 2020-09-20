import pandas as pd


rows = []
df = pd.read_csv('sample_csv.csv')

for _, row in df.iterrows():
    rows.extend([row] * row['Score'])

new_df = pd.DataFrame(rows, columns=df.columns)
new_df.to_csv('bad_csv.csv')
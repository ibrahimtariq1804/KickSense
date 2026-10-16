import pandas as pd

df = pd.read_excel("data/labels/penalty_labels.xlsx")
print("Columns:", df.columns.tolist())
print(f"Total rows: {len(df)}")
print("\nFirst 15 rows:")
print(df.head(15))

import pandas as pd
df = pd.read_csv('data/traind_data.csv')
print(df)
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

print("\nAverage speed:")
print(df["speed"].mean())

print("\nAverage delay:")
print(df["current_delay"].mean())

print("\nAverage travel time:")
print(df["travel_time"].mean())
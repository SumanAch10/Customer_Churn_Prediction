import pandas as pd

df = pd.read_csv("dataset/new_dataset.csv")
first_churned = df[df['Churn'] == 'No'].iloc[0]
print(first_churned)
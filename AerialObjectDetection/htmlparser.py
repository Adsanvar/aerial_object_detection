import pandas as pd

html = pd.read_html('data/Gateway_Village_Estates.html')
df = pd.DataFrame()

for i in html[0]:
    df[i] = i

for p in range(1, len(html)):
    for k in html[p].values:
        df.loc[p] = k

df = df.drop(['Unnamed: 0'], axis=1)
df.to_csv("data/data.csv")


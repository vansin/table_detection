import pandas as pd

df = pd.read_csv('/home/vansin/Nutstore Files/ubuntu/paper/data/trained_origin.csv')


data = pd.pivot_table(df, index=['cmd'], values=['epoch', 'eval_epoch'], )


print(pd)

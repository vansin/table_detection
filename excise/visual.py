import pandas as pd

train_df = pd.read_csv('/home/vansin/Nutstore Files/ubuntu/paper/data/trained_origin.csv')
eval_df = pd.read_csv('/home/vansin/Nutstore Files/ubuntu/paper/data/csv/latest.csv')


train_data = pd.pivot_table(train_df, index=['cmd'], values=[
                      'epoch', 'eval_epoch'], )

eval_data = pd.pivot_table(eval_df, index=['cmd'], values=[
    'epoch', 'eval_epoch'], )


print( train_data_)

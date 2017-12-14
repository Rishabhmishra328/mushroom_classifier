import pandas as pd

# loading data
df_data = pd.read_csv('mushrooms.csv')
    
#creating label assignments
entry_map = {}

# changing labels to numbers
for key in df_data.keys():
    entry_map[key] = df_data[key].unique().tolist()
    
for key in entry_map.keys():
    for value in entry_map[key]: 
        df_data.loc[df_data[key]==value, key] = entry_map[key].index(value)

def map():
    return(entry_map)

def df():
    return (df_data)

def input_layer_length():
    return(input_layer_length)


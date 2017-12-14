import numpy as np
import data_handler as dh
import scipy as sp

# importing data
df_data = dh.df()
dict_map = dh.map()
ill = dh.input_layer_length()
k = 2 #2 clusters for edible and poisnous
convergence_threshold = 10

# inputs
x = df_data.values

# k points
feature_count = len(df_data.columns)
k_points = [np.random.rand(feature_count) for i in range(k)]
point_bag = [[] for i in range(k)]

def train():
    dist_euc = [[] for i in range(k)]
    for point in x:
        for k in range(k):
            dist_euc[k] = sp.spatial.distance.euclidean(k_points[k], point)
        


# k_points = [np]

# for row in df_data.iterrows():
#         print row
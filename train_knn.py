import numpy as np
import data_handler as dh

# importing data
df_data = dh.df()
dict_map = dh.map()
ill = dh.input_layer_length()
k = 2 #2 clusters for edible and poisnous
convergence_threshold = 10

# inputs
x = df_data.values
header = df_data.columns

# k points
feature_count = len(df_data.columns)
k_points = [np.random.rand(feature_count) for i in range(k)]
point_bag = [[]] * k

def train(k_points):

    error = 1000000000000000000000000000
    iteration_counter = 1
    while(error > 0.00000005):
        error,k_points = iteration(x,k_points)
        print('Iteration : %d \n Error : %d' %(iteration_counter,error))
        iteration_counter += 1

    

def iteration(x,k_points):
    dist_euc = [None] * k
    new_points = np.zeros(feature_count) * k

    for point in x[100:]:
        for k_value in range(k):
            dist_euc[k_value] = np.linalg.norm(k_points[k_value] - point)
        point_bag[np.argmin(dist_euc)].append(point)
    
    for index in range(len(point_bag)):
        new_points[index] =  np.sum(point_bag[index])/len(point_bag[index])
        
    error_dist = abs(np.linalg.norm(np.array(new_points) - np.array(k_points)))
    k_points = new_points
    return(error_dist,k_points)
    
def evaluate(test_data,k_points):
    data = [td[1:] for td in test_data]

    accuracy = 0
    for index in range(k):
        k_points[index] = k_points[index][1:]

    for p_index in range(len(data)):
        k_dist = [0., 0.]
        for index in range(k):
            k_dist[index] = abs(np.linalg.norm(np.array(data[p_index]) - np.array(k_points[index])))
        prediction = np.argmin(k_dist)
        if(prediction == test_data[p_index][0]):
            accuracy += 1

    print('Accuracy : %d' %(accuracy/len(test_data)))


train(k_points)
evaluate(x[:100],k_points)
# k_points = [np]

# for row in df_data.iterrows():
#         print row
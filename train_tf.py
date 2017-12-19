import tensorflow as tf
import pandas as pd
import numpy as np

# params
learning_rate = 0.05
epochs = 150
display_step = 10

#data handler
# loading data
df_data = pd.read_csv('mushrooms.csv')
    
#creating label assignments
entry_map = {}

# changing labels to numbers
for key in df_data.keys():
    entry_map[key] = df_data[key].unique().tolist()

# print(entry_map)
    
for key in entry_map.keys():
    # print(df_data[key])
    input_key_data = []
    # print(df_data[key])
    for data_line in df_data[key]:
        input_key_data = [1. if i == entry_map[key].index(data_line) else 0. for i in range(len(entry_map[key]))]
        print(input_key_data)

    # for value in entry_map[key]: 
    #     print(input_key_data)
    #     # data
    #     df_data.loc[df_data[key]==value, key] = entry_map[key].index(value)

# importing data
train_data, test_data = df_data[102:].drop('class', axis = 1).values.astype(np.float32), df_data[1:101].drop('class', axis = 1).values.astype(np.float32)
# print(np.array(test_data))
train_label_data, test_label_data = df_data.loc[102:, 'class'].values, df_data.loc[1:100, 'class'].values
train_labels, test_labels = np.zeros((len(train_label_data), 2)), np.zeros((len(test_label_data), 2))
label_counter = 0

r = [i for i in range(len(test_label_data))]
c = [i for i in test_label_data]
test_labels[r, c] = 1.

r = [i for i in range(len(train_label_data))]
c = [i for i in train_label_data]
train_labels[r, c] = 1.

def map():
    return(entry_map)

def get_data():

    return (df_data)

def input_layer_length():
    # length of feature minus the label
    return(len(entry_map) - 1)

ill = input_layer_length()

def train():
    x = tf.placeholder(tf.float32, [None, ill])
    y = tf.placeholder(tf.float32, [None, 2])

    #model
    W = tf.Variable(tf.zeros([ill,2]))
    b = tf.Variable(tf.zeros([2]))
    #linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)
    #cross entropy
    cost_function = -tf.reduce_sum(y * tf.log(model))
    #gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
    #initializing
    init = tf.global_variables_initializer()

    #firing up session
    with tf.Session() as sess:
        sess.run(init)
        #training
        for iteration in range (epochs):
            avg_cost = 0.
            #fitting data
            sess.run(optimizer, feed_dict={x: train_data, y: train_labels})
            #calculating total loss
            avg_cost += sess.run(cost_function, feed_dict={x: train_data, y: train_labels})/epochs
            #display logs each iteration
            if iteration % display_step == 0 :
                print('Iteration : ', '%04d' % (iteration + 1), 'cost = ', avg_cost)

        print('Training complete')

        #testing
        # predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1)) 
        #accuracy
        # accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
        # print('Accuracy : ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train()





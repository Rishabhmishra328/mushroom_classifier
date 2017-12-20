import tensorflow as tf
import pandas as pd
import numpy as np

# params
learning_rate = 0.0001
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

data_array_collector = []
test_data_array_collector = []
key_counter = 0
keys_for_data = df_data.keys().tolist()
keys_for_data.remove('class')
for key in keys_for_data:
    if key_counter == 0:
        data_array_collector = [np.array([1. if i == entry_map[key].index(data_line) else 0. for i in range(len(entry_map[key]))]) for data_line in df_data[key][101:]]
        test_data_array_collector = [np.array([1. if i == entry_map[key].index(data_line) else 0. for i in range(len(entry_map[key]))]) for data_line in df_data[key][:100]]
    else:
        entry_array = [np.array([1. if i == entry_map[key].index(data_line) else 0. for i in range(len(entry_map[key]))]) for data_line in df_data[key][101:]]
        data_array_collector = np.c_[data_array_collector, entry_array]
        test_entry_array = [np.array([1. if i == entry_map[key].index(data_line) else 0. for i in range(len(entry_map[key]))]) for data_line in df_data[key][:100]]
        test_data_array_collector = np.c_[test_data_array_collector, test_entry_array]
    key_counter += 1

label_array_collector = np.array([[1. if i == entry_map['class'].index(data_line) else 0. for i in range(2)] for data_line in df_data['class'][101:]])
test_label_array_collector = np.array([[1. if i == entry_map['class'].index(data_line) else 0. for i in range(2)] for data_line in df_data['class'][:100]])

print('input : ', data_array_collector[0])
print('output : ', label_array_collector[0])

print('input : ', test_data_array_collector[0])
print('output : ', test_label_array_collector[0])


def train():
    x = tf.placeholder(tf.float32, [None, data_array_collector.shape[1]])
    y = tf.placeholder(tf.float32, [None, label_array_collector.shape[1]])

    #model
    W = tf.Variable(tf.zeros([data_array_collector.shape[1],label_array_collector.shape[1]]))
    b = tf.Variable(tf.zeros([label_array_collector.shape[1]]))
    #linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)
    #cross entropy
    cost_function = -tf.reduce_sum(y * tf.log(model + 1e-10))
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
            sess.run(optimizer, feed_dict={x: data_array_collector, y: label_array_collector})
            #calculating total loss
            avg_cost += sess.run(cost_function, feed_dict={x: data_array_collector, y: label_array_collector})/epochs
            #display logs each iteration
            if iteration % display_step == 0 :
                print('Iteration : ', '%04d' % (iteration + 1), 'cost = ', avg_cost)

        print('Training complete')

        #testing
        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1)) 
        #accuracy
        accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
        print('Accuracy : ', accuracy.eval({x: test_data_array_collector, y: test_label_array_collector}), 'Predictions : ', predictions.eval({x: test_data_array_collector, y: test_label_array_collector}))

train()





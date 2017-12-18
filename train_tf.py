import tensorflow as tf
import pandas as pd
import numpy as np

#data handler
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
    # length of feature minus the label
    return(len(entry_map) - 1)



# importing data
train_data, test_data = df_data[102:].drop('class', axis = 1).values.astype(np.float32), df_data[1:101].drop('class', axis = 1).values.astype(np.float32)
# print(np.array(test_data))
train_labels, test_labels = df_data.loc[102:, 'class'].values.astype(np.float32), df_data.loc[1:101, 'class'].values.astype(np.float32)
# print(test_labels)

ill = input_layer_length()

# params
learning_rate = 0.05
epochs = 100
batch = 100
display_step = 2

def train():
    x = tf.placeholder(tf.float32,[None,ill])
    y = tf.placeholder(tf.float32,[None,2])

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
            x, y = train_data, train_labels
            # print(str(train_data))
            # print(train_labels)
            total_batch = (int)(x.size / (batch * 22))
            print(total_batch)
            batch_index = 0

            #looping over batches
            for i in range(total_batch):
                # print(x[batch_index:(batch_index + batch)])
                batch_x, batch_y = tf.convert_to_tensor(x[batch_index:(batch_index + batch)]), tf.convert_to_tensor(y[batch_index:(batch_index + batch)])
                print(type(batch_x)clear
                    , batch_y)
                batch_index += batch
                #fitting data
                sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
                #calculating total loss
                avg_cost += sess.run(cost_function, feed_dict = {x: batch_x, y: batch_y})/total_batch
            #display logs each iteration
            if iteration % display_step == 0 :
                print('Iteration : ', '%04d' % (iteration + 1), 'cost = ', '{:.9f}'.format(avg_cost))

        print('Training complete')

        #testing
        # predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1)) 
        #accuracy
        # accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
        # print('Accuracy : ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train()





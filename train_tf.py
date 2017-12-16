import tensorflow as tf
import data_handler as dh

# importing data
csv_data = dh.df()
dict_map = dh.map()
train_data, test_data = dh.network_data()
ill = dh.input_layer_length()

# params
learning_rate = 0.05
epochs = 100
batch = 50
display_step = 2

def train():
    x = tf.placeholder(tf.float32,[None,ill])
    y = tf.placeholder9tf.float32,[None,2]

    #model
    W = tf.Variable(tf.zeros([ill,2]))
    b = tf.Variable(tf.zeros([2]))


    with tf.name_scope('Wx_b') as scope:
        #linear model
        model = tf.nn.softmax(tf.matmul(x, W) + b)


    with tf.name_scope('cost_function') as scope:
        #cross entropy
        cost_function = -tf.reduce_sum(y * tf.log(model))

    with tf.name_scope('train') as scope:
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
            total_batch = len(train_data)/batch
            #looping over batches
            for i in range(total_batch):
                batch_x, batch_y = get_next_batch(batch)
                #fitting data
                sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
                #calculating total loss
                avg_cost += sess.run(cost_function, feed_dict = {x: batch_x, y: batch_y})/total_batch
            #display logs each iteration
            if iteration % display_step == 0 :
                print('Iteration : ', '%04d' % (iteration + 1), 'cost = ', '{:.9f}'.format(avg_cost))

        print('Training complete')

        #testing
        predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1)) 
        #accuracy
        accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
        print('Accuracy : ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train()





import tensorflow as tf
import data_handler as dh

# importing data
csv_data = dh.csv()
dict_map = dh.map()
ill = dh.input_layer_length()

# params
learning_rate = 0.05
epochs = 100
batch = 50
display_step = 2

def train():
    x = tf.placeholder(tf.float32,[None,input_layer_length])
    y = tf.placeholder9tf.float32,[None,2]

    # model
    W = tf.Variable(tf.zeros([input_layer_length,2]))
    b = tf.Variable(tf.zeros([2]))


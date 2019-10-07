import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# import data
df = pd.read_csv('data_stocks.csv')
#print(df.head())

# drop date variable
df = df.drop(['DATE'], 1)

# dimensions of dataset
n = df.shape[0]
p = df.shape[1]

# make data a numpy array
df = df.values

#DISPLAY S&P500 PLOT
#plt.plot(df[:,0]) #plot first column which is SP500
#plt.show()

# training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
# arrange test & train data evenly between start and end values
# train data makes up 80% of data and is followed by the 20% of test data
data_train = df[np.arange(train_start, train_end), :]
data_test = df[np.arange(test_start, test_end), :]

# scale data
# input is scaled to simplify activation function
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

# build x and y
x_train = data_train[:, 1:]
y_train = data_train[:, 0]
x_test = data_test[:, 1:]
y_test = data_test[:, 0]

# ------- tensorflow example to add a and b -------

# define a and b as placeholders (input placeholders)
# a = tf.placeholder(dtype=tf.int8)
# b = tf.placeholder(dtype=tf.int8)

# while placeholders are used to store input and target data in the graph, variables are used as flexible containers within the graph that are allowed to change during graph execution
# weights and biases are represented as variables in order to adapt during training

# define the addition
# c = tf.add(a, b)

# initialize and run the graph
# graph = tf.Session()
# graph.run(c, feed_dict={a: 5, b: 4})

# ------- tensorflow for stock prediction -------

# model architecture parameters
n_stocks = 500
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1

# initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# None argument indicates that we do not yet know the number of observations that flow
# through the neural net in each batch
x = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
y = tf.placeholder(dtype=tf.float32, shape=[None])

# the model consists of four hidden layers. the first layer contains 1024 neurons. subsequent hidden layers are always half the size of the previous layer, do the fourth hidden layer is 128 neurons. this neuron amount reduction compresses the information the network identifies in the previous layers.

# layer 1: variables for hidden weights and biases
w_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# layer 2: variables for hidden weights and biases
w_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# layer 3: variables for hidden weights and biases
w_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

# layer 4: variables for hidden weights and biases
w_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# output layer: variables for output weights and biases
w_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# hidden layers are transformed by activation functions (which introduce non-linearity)
hidden_1 = tf.nn.relu(tf.add(tf.matmul(x, w_hidden_1),
                             bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, w_hidden_2),
                             bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, w_hidden_3),
                             bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, w_hidden_4),
                             bias_hidden_4))

# output layer (must be transposed)

out = tf.transpose(tf.add(tf.matmul(hidden_4, w_out),
                          bias_out))

# cost function
mse = tf.reduce_mean(tf.squared_difference(out, y))

# optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

print("3");

# make session
net = tf.Session()

# run initializer
net.run(tf.global_variables_initializer())

print("2");

# setup interactive plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test*0.5)
plt.show()

# number of epochs and batch size
epochs = 10
batch_size = 256

print("1");

for e in range(epochs):
    # shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = x_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # run optimizer with batch
        net.run(opt, feed_dict={x: batch_x, y: batch_y})

        # show progress
        if np.mod(i, 5) == 0:
            # Prediction
            pred = net.run(out, feed_dict={x: x_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
            plt.savefig(file_name)
            plt.pause(0.01)

# print final mse after training            
mse_final = net.run(mse, feed_dict={x: x_test, y: y_test})
print(mse_final)

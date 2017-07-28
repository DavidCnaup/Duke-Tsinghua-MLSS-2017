import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Helper functions for creating weight variables
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolutional neural network functions
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Model Inputs
x = tf.placeholder("float", [None, 784])  ### MNIST images enter graph here ###
y_ = tf.placeholder("float", [None, 10])  ### MNIST labels enter graph here ###
# Define the graph


### Create your CNN here##
### Make sure to name your CNN output as y_conv ###
input_image = tf.reshape(x, [-1, 28, 28, 1])  # reshape input image
W_conv1 = weight_variable([5, 5, 1, 32])  # conv1 layer
b_conv1 = bias_variable([32])
feature_conv1 = tf.nn.relu(conv2d(input_image, W_conv1) + b_conv1)
feature_pool1 = max_pool_2x2(feature_conv1)  # maxpooling1 layer
W_conv2 = weight_variable([5, 5, 32, 64])  # conv2 layer
b_conv2 = bias_variable([64])
feature_conv2 = tf.nn.relu(conv2d(feature_pool1, W_conv2) + b_conv2)
feature_pool2 = max_pool_2x2(feature_conv2)  # maxpooling2 layer
W_fc1 = weight_variable([49 * 64, 1024])  # fc1 layer
b_fc1 = bias_variable([1024])
fc_flat = tf.reshape(feature_pool2, [-1, 49 * 64])
output_fc1 = tf.nn.relu(tf.matmul(fc_flat, W_fc1) + b_fc1)
W_fc2 = weight_variable([1024, 10])  # fc2 layer
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(output_fc1, W_fc2) + b_fc2)
# Loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Evaluation
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Training regimen
    for i in range(10000):
        # Validate every 250th batch
        if i % 250 == 0:
            validation_accuracy = 0
            for v in range(10):
                batch = mnist.validation.next_batch(50)
                validation_accuracy += (1 / 10) * accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print('step %d, validation accuracy %g' % (i, validation_accuracy))

        # Train
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
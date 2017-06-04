from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sess = tf.InteractiveSession()

#settings
LEARNING_RATE = 1e-4
TRAINING_ITERATIONS = 20000
DROPOUT = 0.5
BATCH_SIZE = 50
VALIDATION_SIZE = 2000
IMAGE_TO_DISPLAY = 10

#data preparation
data = pd.read_csv()

images = data.iloc[:,1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0/255.0)

image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

def display(img):
    one_image = img.reshape(image_width, image_height)
    plt.axis = 'off'
    plt.imshow(one_image, cmap=cm.binary)

display(images[IMAGE_TO_DISPLAY])

labels_flat = data[[0]].values.ravel()
labels_count = np.unique(labels_flat).shape[0]

def dense_to_onehot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    labels_onehot.flat[index_offset + labels_dense.ravel()] = 1

labels = dense_to_onehot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

#network
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, image_size])
y_ = tf.placeholder(tf.float32, shape=[None, labels_count])

#FIRST CONVOLUTIONAL LAYER
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,image_width,image_height,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4, 8))
layer1 = tf.transpose(layer1, (0,3,1,4,2))
layer1 = tf.reshape(layer1, (-1,image_height*4, image_width*8))

#SECOND CONVOLUTIONAL LAYER
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

layer2 = tf.reshape(h_conv2, (-1,14,14,4,16))
layer2 = tf.transpose(layer2, (0,3,1,4,2))
layer2 = tf.reshape(layer2, (-1,14*4,14*16))

#FULLY CONNECTED LAYER
W_fcl = weight_variable([7*7*64,1024])
b_fcl = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat, W_fcl) + b_fcl)

#DROPOUT
keep_prob = tf.placeholder(tf.float32)
h_fcl_drop = tf.nn.dropout(h_fcl, keep_prob)

#READOUT LAYER
W_fc2 = weight_variable([1024,labels_count])
b_fc2 = bias_variable([labels_count])

y_conv = tf.nn.softmax(tf.matmul(h_fcl_drop, W_fc2) + b_fc2)

#TRAINING AND EVALUATING
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_predition = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
predict = tf.argmax(y,1)
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

sess.run(tf.global_variables_initializer

train_accuracies = []
validation_accuracies = []
x_range = []
display_step = 1

for i in range(TRAINING_ITERATIONS):
    batch_xs, batch_ys = next_batch(BATCH_SIZE)

    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs,
                                                  y:batch_ys,
                                                  keep_prob=1.0})
    if(VALIDATION_SIZE):
        validation_accuracy = accuracy.eval(feed_dict={x:validation_images[0:BATCH_SIZE],
                                                       y:validation_labels[0:BATCH_SIZE],
                                                       keep_prob:1.0})
        print{'training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i)}
        validation_accuracies.append(train_accuracy)
    else:
        print('training accuracy => %.4f for step %d'%(train_accuracy, i))
    train_accuracies.append(train_accuracy)
    x_range.append(i)
    if i%(display_step*10) == 0 and i:
        display_step *= 10

sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys,keep_prob=DROPOUT})

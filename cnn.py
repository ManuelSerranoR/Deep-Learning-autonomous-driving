from __future__ import division, print_function, absolute_import
print('Importing libraries...')
import numpy as np
import imageio
import math
import pandas as pd
import tensorflow as tf
from PIL import Image
from skimage import transform #For downsizing images
from sklearn.model_selection import train_test_split
print('Done')


#----------------------------------------------------------------------------------------------------------------
#--------------------------------------------------DATA PREPARATION----------------------------------------------
#----------------------------------------------------------------------------------------------------------------

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


size_of_dataSet = 15000
percentage_train_test = 0.8

indexes = np.arange(size_of_dataSet)
np.random.shuffle(indexes)

ind_train = indexes[:int(len(indexes)*percentage_train_test)]
ind_test = indexes[int(len(indexes)*percentage_train_test):]

#Calculate mean RGB for preprocessing
rgb_sum = 0
for image_path in range(size_of_dataSet):
    rgb_sum += np.mean(np.asarray(imageio.imread('dataX/'+str(image_path)+'.png')), axis=(0,1))
mean_image = rgb_sum/size_of_dataSet


# We do load all Y, but not X due to computational workload
dataY = pd.read_csv('dataY.csv', index_col=0)
dataY = dataY[['trackPos']].as_matrix()

# Normalize parameters around [0.1,1]
#dataY[:,0] = scale_range(dataY[:,0],0,1)
#dataY = dataY*(180/math.pi) #To make it degrees
#dataY[:,1] = scale_range(dataY[:,1],0.1,1)
dataY = dataY*8 #To make it meters
#dataY[:,2] = scale_range(dataY[:,2],0.1,1)


#----------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------CNN-----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

# Training Parameters
learning_rate = 0.001
num_steps = 5000
batch_size = 64
display_step = 10

# Network Parameters
num_outputs = 1 # angle
                                                                            
# tf Graph input
X = tf.placeholder(tf.float32, [None, 224, 224, 3], name="X")
Y = tf.placeholder(tf.float32, [None, num_outputs], name="Y")
dropout_prob = tf.placeholder_with_default(1.0, shape=())

# Create model
def conv_net(x):

    # --------------Convolution Layers--------------
    conv0 = tf.layers.conv2d(inputs=x, filters = 16, kernel_size = 3, strides = (1, 1), activation=tf.nn.relu, padding='same')
    conv1 = tf.layers.conv2d(inputs=conv0, filters = 32, kernel_size = 3, strides = (1, 1), activation=tf.nn.relu, padding='same')
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=1e-05, beta=0.75, bias=1.0)
    pool1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters = 32, kernel_size = 3, strides = (1, 1), activation=tf.nn.relu, padding='same')
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=1e-05, beta=0.75, bias=1.0)
    pool2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters = 32, kernel_size = 3, strides = (1, 1), activation=tf.nn.relu, padding='same')
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(inputs=pool3, filters = 64, kernel_size = 3, strides = (1, 1), activation=tf.nn.relu, padding='same')
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    #Reshape conv3 output to fit fully connected layer input
    shape = int(np.prod(pool4.get_shape()[1:]))
    pool4_flat = tf.reshape(pool4, [-1, shape])

    # --------------Fully connected layer--------------
    dense1 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
    dropout1 = tf.nn.dropout(dense1, dropout_prob)
    dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)
    dropout2 = tf.nn.dropout(dense2, dropout_prob)
    dense3 = tf.layers.dense(inputs=dropout2, units=256, activation=tf.nn.relu)
    dropout3 = tf.nn.dropout(dense3, dropout_prob)
    dense4 = tf.layers.dense(inputs=dropout3, units=num_outputs)

    return dense4

# Construct model_selection
predicted_output = conv_net(X)
#shapeof = tf.shape(predicted_output)
predicted_output_tensor = tf.convert_to_tensor(predicted_output, name = "output")

#Define loss and optimizer
loss_op = tf.nn.l2_loss(Y - predicted_output_tensor, name = "Loss")
#loss_op = tf.losses.absolute_difference(Y, predicted_output_tensor)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op, name = "train")


#Data to transfer
tf.GraphKeys.USEFUL = 'useful'
tf.add_to_collection(tf.GraphKeys.USEFUL, X)
tf.add_to_collection(tf.GraphKeys.USEFUL, Y)
tf.add_to_collection(tf.GraphKeys.USEFUL, loss_op)
tf.add_to_collection(tf.GraphKeys.USEFUL, predicted_output_tensor)
tf.add_to_collection(tf.GraphKeys.USEFUL, train_op)


# 'Saver' saves and restores all the variables
saver = tf.train.Saver()

train_loss = []
test_loss = []

# Start training
print('Start training')
with tf.Session() as sess:

    ## Initialize the variables (i.e. assign their default value)
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    for step in range(1, num_steps + 1):

        # Batch train preparation
        random_indexes_train = np.random.choice(ind_train, batch_size)
        count_train = 0
        batch_X_train = np.empty((batch_size, 224, 224, 3))
        for image_path in random_indexes_train:
            batch_X_train[count_train] = np.asarray(imageio.imread('dataX/'+str(image_path)+'.png')) - mean_image
            count_train += 1
        batch_Y_train = dataY[random_indexes_train]

        #Run optimization (backpropagation)
        sess.run(train_op, feed_dict = {X: batch_X_train, Y: batch_Y_train, dropout_prob: 0.5})

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            print('Step', step)
            loss_train = sess.run(loss_op, feed_dict = {X: batch_X_train, Y: batch_Y_train})/len(batch_X_train)
            train_loss.append(loss_train)
            print('Train loss is', loss_train)
            #print(sess.run(shapeof, feed_dict = {X: batch_X_train, Y: batch_Y_train}))

            # Batch test preparation
            random_indexes_test = np.random.choice(ind_test, 32)
            count_test = 0
            batch_X_test = np.empty((32, 224, 224, 3))
            for image_path in random_indexes_test:
                batch_X_test[count_test] = np.asarray(imageio.imread('dataX/'+str(image_path)+'.png')) - mean_image
                count_test += 1
            batch_Y_test = dataY[random_indexes_test]

            loss_test = sess.run(loss_op, feed_dict={X: batch_X_test, Y: batch_Y_test})/32
            test_loss.append(loss_test)
            print('Test loss is', loss_test)
            print('Prediction:', sess.run(predicted_output, feed_dict = {X: batch_X_test, Y: batch_Y_test})[0])
            print('Ground truth:', batch_Y_test[0], '\n')
        

    print("Optimization Finished!")

    save_path = saver.save(sess, "my_test_model")
    print("Model saved in path: %s" % save_path)

collected_losses = pd.DataFrame(np.array([np.array(train_loss), np.array(test_loss)]).transpose(), columns=["train_loss","test_loss"])
collected_losses.to_csv("collected_losses.csv")

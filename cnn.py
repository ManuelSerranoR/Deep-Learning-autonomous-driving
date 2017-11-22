from __future__ import division, print_function, absolute_import
print('Importing libraries...')
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from skimage import transform #For downsizing images
from sklearn.model_selection import train_test_split
print('Done')

#Open images from pickle
print('--------------CREATING FEATURES DATASET--------------')
print('Unpickling images...')
with open ('../Data/CG Speedway number 1/train_set_c1', 'rb') as fp:
    X1 = pickle.load(fp)

with open ('../Data/CG Speedway number 1/train_set_c2', 'rb') as fp:
    X2 = pickle.load(fp)
    
with open ('../Data/CG Speedway number 1/train_set_c3', 'rb') as fp:
    X3 = pickle.load(fp)
    
with open ('../Data/CG Speedway number 1/train_set_c4', 'rb') as fp:
    X4 = pickle.load(fp)

if not (len(X1)==len(X2)==len(X3)): 
    print('Inconsistent data images')
else: print('Done')
num_images = len(X1)

#Downsample images from 640x480 to 280x210 and store them in training_images
training_images1, training_images2, training_images3, training_images4 = [], [], [], []
print('Resizing images...')
for i in range(num_images):
    training_images1.append(transform.resize(X1[i][0], (210, 280, 3), order=0))
    training_images2.append(transform.resize(X2[i][0], (210, 280, 3), order=0))
    training_images3.append(transform.resize(X3[i][0], (210, 280, 3), order=0))
    training_images4.append(transform.resize(X4[i][0], (210, 280, 3), order=0))
print('Done')
#plt.imshow(training_images1[0])
#plt.show()
print('Creating final feature dataset...')
#dataX = np.array(training_images1)
dataX = np.concatenate((training_images1, training_images2), axis=0)
dataX = np.concatenate((dataX, training_images3), axis=0)
dataX = np.concatenate((dataX, training_images4), axis=0)
print('Converting into grayscale...')
dataX = np.dot(dataX[...,:3], [0.114, 0.587, 0.299])
print('Features dataset finished with shape', dataX.shape)


print('--------------CREATING LABELS DATASET--------------')
print('Unpickling driving parameters...')
with open ('../Data/CG Speedway number 1/obs_c1', 'rb') as fp:
    driving_parameters1 = pickle.load(fp)
with open ('../Data/CG Speedway number 1/obs_c2', 'rb') as fp:
    driving_parameters2 = pickle.load(fp)
with open ('../Data/CG Speedway number 1/obs_c3', 'rb') as fp:
    driving_parameters3 = pickle.load(fp)
with open ('../Data/CG Speedway number 1/obs_c4', 'rb') as fp:
    driving_parameters4 = pickle.load(fp)

if not (len(driving_parameters1)==len(driving_parameters2)==len(driving_parameters3)==len(driving_parameters4)): 
    print('Inconsistent data labels')
else: print('Done')

print('Concatentating all driving parameter sets')
num_param = len(driving_parameters1)
angle = np.array([o[3] for o in driving_parameters1])
angle = np.concatenate((angle, np.array([o[3] for o in driving_parameters2])))
angle = np.concatenate((angle, np.array([o[3] for o in driving_parameters3])))
angle = np.concatenate((angle, np.array([o[3] for o in driving_parameters4])))

radius = np.array([o[4] for o in driving_parameters1])
radius = np.concatenate((radius, np.array([o[4] for o in driving_parameters2])))
radius = np.concatenate((radius, np.array([o[4] for o in driving_parameters3])))
radius = np.concatenate((radius, np.array([o[4] for o in driving_parameters4])))

trackPos = np.array([o[6] for o in driving_parameters1])
trackPos = np.concatenate((trackPos, np.array([o[6] for o in driving_parameters2])))
trackPos = np.concatenate((trackPos, np.array([o[6] for o in driving_parameters3])))
trackPos = np.concatenate((trackPos, np.array([o[6] for o in driving_parameters4])))

dataY = pd.DataFrame(np.array([trackPos, radius, angle]).transpose(),
                    columns=["trackPos","radius", "angle"])

print('Labels dataset finished with shape', dataY.shape)

#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#------------FROM HERE, ALL PREVIOUS CODE WAS SOLELY RELATED TO DATA CLEANING, NOT WE BUILD THE CNN--------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.10)

# Training Parameters
learning_rate = 0.001
num_steps = 1600
batch_size = 100
display_step = 10

# Network Parameters
num_outputs = 3 # For now we will predict trackPos, radius, angle
                                                                            
# tf Graph input
X = tf.placeholder(tf.float32, [None, 210, 280, 3], name="X")
Y = tf.placeholder(tf.float32, [None, num_outputs], name="Y")

# Create model
#def conv_net(x, weights, biases):
def conv_net(x):

    # --------------Convolution Layers--------------
    conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, strides=(2,2), activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)
 
    #Reshape conv3 output to fit fully connected layer input
    conv3 = tf.reshape(conv3, [-1, 25*34*64])

    # --------------Fully connected layer--------------
    dense1 = tf.layers.dense(inputs=conv3, units=1024, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=3)
    return dense3


    # Construct model_selection
predicted_output = conv_net(X)  #, weights, biases)
predicted_output_tensor = tf.convert_to_tensor(predicted_output, name="output")

#Define loss and optimizer
loss_op = tf.nn.l2_loss(Y-predicted_output_tensor, name = "Loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


#Data to transfer
tf.GraphKeys.USEFUL = 'useful'
tf.add_to_collection(tf.GraphKeys.USEFUL, X)
tf.add_to_collection(tf.GraphKeys.USEFUL, Y)
tf.add_to_collection(tf.GraphKeys.USEFUL, loss_op)
tf.add_to_collection(tf.GraphKeys.USEFUL, predicted_output_tensor)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# 'Saver' saves and restores all the variables
saver = tf.train.Saver()

train_loss = []
test_loss = []

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        # Run optimization op (backprop)
        random_indexes = np.random.choice(X_train.shape[0], batch_size)
        batch_X = dataX[random_indexes]
        batch_Y = dataY.iloc[random_indexes]
        #Run optimization (backpropagation)
        sess.run(train_op, feed_dict={X: batch_X, Y: batch_Y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss_train = sess.run(loss_op, feed_dict={X: batch_X, Y: batch_Y})/len(batch_X)
            train_loss.append(loss_train)
            print('Train loss at step ', step, ' is ', loss_train)
            loss_test = sess.run(loss_op, feed_dict={X: X_test, Y: Y_test})/len(X_test)
            test_loss.append(loss_test)
            print('Test loss at step ', step, ' is ', loss_test)

    print("Optimization 1 Finished!")

    save_path = saver.save(sess, "my_test_model")
    print("Model saved in path: %s" % save_path)

collected_losses = pd.DataFrame(np.array([np.array(train_loss), np.array(test_loss)]).transpose(), columns=["train_loss","test_loss"])
collected_losses.to_csv("collected_losses.csv")

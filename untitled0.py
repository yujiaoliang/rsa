# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:26:08 2019

@author: yujiaoliang
"""

""" Neural Network with Eager API.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow's Eager API. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
#from __future__ import print_function

import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeavePOut

# Set Eager API
#tf.enable_eager_execution()
#tfe = tf.contrib.eager

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
x = []
y = []
with open (r"C:\\Users\\yujiaoliang\\Desktop\\new\\two.csv") as csvfile1:
    csv_reader = csv.reader(csvfile1)
    for row in csv_reader:
        x.append(row)
with open (r"C:\\Users\\yujiaoliang\\Desktop\\new\\output.csv") as csvfile2:
    csv_reader = csv.reader(csvfile2)
    for row in csv_reader:
        y.append(row)

#x= np.array(x).astype(np.float32)
#y = np.array(y).astype(np.int32)
# =============================================================================
# x_qtrain = []
# x_qtest = []
# y_qtrain =[]
# y_qtest =[]
# lpo=LeavePOut(p=20)
# lpo.get_n_splits(x)
# for train_index,test_index in lpo.split(x,y):
#     x_train_i,x_test_i=x[train_index],x[test_index]
#     y_train_i,y_test_i=y[train_index],y[test_index]
#     x_qtrain.append(x_train_i)
#     x_qtest.append(x_test_i)
#     y_qtrain.append(y_train_i)
#     y_qtest.append(y_test_i)
# 
# x_train = x_qtrain
# x_test = x_qtest
# y_train =y_qtrain
# y_test =y_qtest
# =============================================================================
    
x = np.array(x).astype(np.float32)
y = np.array(y).astype(np.int32)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=0)
#y = np.zeros((300,11))
#for i in range(300):
#    k = ys[i]
#    y[i][k] = 1
#y = np.array(y).astype(np.float32)

tf_x = tf.placeholder(tf.float32, x_train.shape)     # input x
tf_y = tf.placeholder(tf.int32, y_train.shape)     # input y
tf_is_training = tf.placeholder(tf.bool,None)
# create dataloader    
dataset = tf.data.Dataset.from_tensor_slices((tf_x, tf_y))
dataset = dataset.shuffle(buffer_size=1000)   # choose data randomly from this buffer
dataset = dataset.batch(128)                   # batch size you will use
dataset = dataset.repeat()                   # repeat for 3 epochs
iterator = dataset.make_initializable_iterator()  # later we have to initialize this one


bx, by = iterator.get_next()
l1 = tf.layers.dense(bx, 256, tf.nn.relu)          # hidden layer
l1 = tf.layers.dropout(l1,rate=0,training = tf_is_training)
output = tf.layers.dense(l1, 64, )                     # output layer



#loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) # loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=by, logits=output)           # compute cost
accuracy = tf.metrics.accuracy(  labels=by, predictions=tf.argmax(output, axis=1),)[1]         # return (acc, update_op), and create 2 local variables

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                                                 # control training and others
#init_op = tf.group(tf.global_variables_initializer())
sess.run([iterator.initializer, tf.global_variables_initializer(),tf.local_variables_initializer()],feed_dict={tf_x: x_train, tf_y:y_train})     # initialize var in graph

draw_s= []
draw_l =[]
draw_a =[]
plt.ion()   # something about plotting
for step in range(20000):
  try:
    _, trainl,accu1 = sess.run([train_op, loss,accuracy],{ tf_is_training:True})                      
    if step % 1000 == 0:
        draw_s.append(step)
        draw_l.append(trainl)
        draw_a.append(accu1)
        accu = sess.run(accuracy, {bx: x_test, by: y_test,tf_is_training:False})    # test
#        print('step: %i/200' % step, '|train loss:', trainl, '|test acc:', accu)
  except tf.errors.OutOfRangeError:     # if training takes more than 3 epochs, training will be stopped
    print('Finish the last epoch.')
    break
plt.cla()
plt.plot(draw_s,draw_l,ls="-")
plt.xlabel("step")
plt.ylabel("loss")
plt.ioff()
plt.show()
plt.ion() 
plt.plot(draw_s,draw_a,ls="-")
plt.xlabel("step")
plt.ylabel("accurancy")
plt.ioff()
plt.show()
#    except tf.errors.OutOfRangeError:     # if training takes more than 3 epochs, training will be stopped
#         print('Finish the last epoch.')
#         break
#plt.ioff()
#plt.show()
#plt.ion()   # something about plotting
#for step in range(100):
#    # train and net output
#    _, acc, pred = sess.run([train_op, accuracy, output], {tf_x: x, tf_y: y})
#    if step % 2 == 0:
#        # plot and show learning process
#        plt.cla()
#        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
#        plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 20, 'color': 'red'})
#        plt.pause(0.1)
#
#plt.ioff()
#plt.show()
# Parameters
# =============================================================================
# learning_rate = 0.001
# num_steps = 1000
# batch_size = 128
# display_step = 100
# 
# # Network Parameters
# n_hidden_1 = 256 # 1st layer number of neurons
# n_hidden_2 = 256 # 2nd layer number of neurons
# num_input = 48 # MNIST data input (img shape: 28*28)
# num_classes = 11 # MNIST total classes (0-9 digits)
# 
# # Using TF Dataset to split data into batches
# dataset = tf.data.Dataset.from_tensor_slices((x, ys))
# dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
# dataset_iter = tfe.Iterator(dataset)
# 
# 
# # Define the neural network. To use eager API and tf.layers API together,
# # we must instantiate a tfe.Network class as follow:
# class NeuralNet(tfe.Network):
#     def __init__(self):
#         # Define each layer
#         super(NeuralNet, self).__init__()
#         # Hidden fully connected layer with 256 neurons
#         self.layer1 = self.track_layer(
#             tf.layers.Dense(n_hidden_1, activation=tf.nn.relu))
#         # Hidden fully connected layer with 256 neurons
#         self.layer2 = self.track_layer(
#             tf.layers.Dense(n_hidden_2, activation=tf.nn.relu))
#         # Output fully connected layer with a neuron for each class
#         self.out_layer = self.track_layer(tf.layers.Dense(num_classes))
# 
#     def call(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         return self.out_layer(x)
# 
# 
# neural_net = NeuralNet()
# 
# 
# # Cross-Entropy loss function
# def loss_fn(inference_fn, inputs, labels):
#     # Using sparse_softmax cross entropy
#     return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits=inference_fn(inputs), labels=ys))
# 
#  
# # Calculate accuracy
# def accuracy_fn(inference_fn, inputs, labels):
#     prediction = tf.nn.softmax(inference_fn(inputs))
#     correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
#     return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 
# 
# # SGD Optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# # Compute gradients
# grad = tfe.implicit_gradients(loss_fn)
# 
# # Training
# average_loss = 0.
# average_acc = 0.
# for step in range(num_steps):
# 
#     # Iterate through the dataset
#     d = dataset_iter.next()
# 
#     # Images
#     x_batch = d[0]
#     # Labels
#     y_batch = tf.cast(d[1], dtype=tf.int64)
# 
#     # Compute the batch loss
#     batch_loss = loss_fn(neural_net, x_batch, y_batch)
#     average_loss += batch_loss
#     # Compute the batch accuracy
#     batch_accuracy = accuracy_fn(neural_net, x_batch, y_batch)
#     average_acc += batch_accuracy
# 
#     if step == 0:
#         # Display the initial cost, before optimizing
#         print("Initial loss= {:.9f}".format(average_loss))
# 
#     # Update the variables following gradients info
#     optimizer.apply_gradients(grad(neural_net, x_batch, y_batch))
# 
#     # Display info
#     if (step + 1) % display_step == 0 or step == 0:
#         if step > 0:
#             average_loss /= display_step
#             average_acc /= display_step
#         print("Step:", '%04d' % (step + 1), " loss=",
#               "{:.9f}".format(average_loss), " accuracy=",
#               "{:.4f}".format(average_acc))
#         average_loss = 0.
#         average_acc = 0.
# 
# =============================================================================
# Evaluate model on the test image set
#testX = mnist.test.images
#testY = mnist.test.labels
#
#test_acc = accuracy_fn(neural_net, testX, testY)
#print("Testset Accuracy: {:.4f}".format(test_acc))


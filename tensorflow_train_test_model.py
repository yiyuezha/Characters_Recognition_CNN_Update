# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
import numpy as np
import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 32
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 63
VALIDATION_SIZE = 15*63  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 1000
EVAL_BATCH_SIZE = 945
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
OUTPUT_CHANNELS1=16
OUTPUT_CHANNELS2=32
MIDDLE_LAYER=128
FLAGS = None


def error_rate(predictions, labels):
  
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def main(_):
  
    # Get the data
  train_data = numpy.loadtxt('train_data.txt')
  train_labels = numpy.loadtxt('train_label.txt')
  test_data = numpy.loadtxt('test_data.txt')
  #test_data = test_data[0:1,:]
  test_labels = numpy.loadtxt('test_label.txt')
  #test_labels = test_labels[0:1,:]
  print (test_labels)
  test_labels = np.argmax(test_labels,1)
  print (train_data.shape)
  # Generate a validation set.
  #validation_data = train_data[:VALIDATION_SIZE, ...]
  #validation_labels = train_labels[:VALIDATION_SIZE]
  #train_data = train_data[VALIDATION_SIZE:, ...]
  #train_labels = train_labels[VALIDATION_SIZE:]
  num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.global_variables_initializer().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, OUTPUT_CHANNELS1],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=tf.float32))
  conv1_biases = tf.Variable(tf.zeros([OUTPUT_CHANNELS1], dtype=tf.float32))
  conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, OUTPUT_CHANNELS1, OUTPUT_CHANNELS2],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=tf.float32))
  conv2_biases = tf.Variable(tf.zeros([OUTPUT_CHANNELS2], dtype=tf.float32))
 
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([2048, MIDDLE_LAYER],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[MIDDLE_LAYER], dtype=tf.float32))
  fc2_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([MIDDLE_LAYER, 63],
                          stddev=0.1,
                          seed=SEED,
                          dtype=tf.float32))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[63], dtype=tf.float32))
  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv1 = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool1 = tf.nn.max_pool(relu1,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    #Drop Out
    if train:
      pool1 = tf.nn.dropout(pool1, 0.5, seed=SEED)
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    

    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv2 = tf.nn.conv2d(pool1,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool2 = tf.nn.max_pool(relu2,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    #Drop Out
    if train:
      pool2 = tf.nn.dropout(pool2, 0.5, seed=SEED)
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.


    pool_shape = pool2.get_shape().as_list()
    reshape = tf.reshape(
        pool2,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    
    #print pool_shape
    #print reshape.get_shape().as_list()
    fc1=tf.matmul(reshape, fc1_weights) + fc1_biases
    fc1=tf.nn.relu(fc1)
    if train:
      fc1 = tf.nn.dropout(fc1, 0.5, seed=SEED)
    fc2=tf.matmul(fc1, fc2_weights) + fc2_biases
    return fc2

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels_node, logits=logits))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0, dtype=tf.float32)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  lr_adam=0.001
  # Use simple momentum for the optimization.
  optimizer = tf.train.AdamOptimizer(lr_adam).minimize(loss)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    #print data.shape
    test_data=data
    test_data=test_data.reshape([945,32,32,1])
    predictions[:, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: test_data})
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
  #    print train_data.shape
  #    print offset
  #    print "batch_data"+str(batch_data.shape)
      batch_data=batch_data.reshape([BATCH_SIZE,32,32,1])
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    #  print batch_labels
      batch_labels = np.argmax(batch_labels,1)
    #  print batch_labels
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the optimizer to update weights.
      sess.run(optimizer, feed_dict=feed_dict)

      
      # print some extra information once reach the evaluation frequency
      va = open("error.txt","w")
      if step % EVAL_FREQUENCY == 0:
        # fetch some extra nodes' data
        l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                      feed_dict=feed_dict)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(test_data, sess), test_labels))
        sys.stdout.flush()
        va.write(str(error_rate(
             eval_in_batches(test_data, sess), test_labels)))
        va.write(" " + str(step))
        va.write("\n")
    # Finally print the result!
    #np.save("weight_save/conv1_w.npy",sess.run(conv1_weights)) 
  	
        if (step == 14600 ):
            conv1_weights_v=sess.run(conv1_weights)
            #print conv1_weights_v.shape
            f=open("weight_save/conv1_weights.txt","w")
            for i in range(OUTPUT_CHANNELS1):
                for j in range(1):
                    for k in range(5):
                        for z in range(5):
                            f.write(str(conv1_weights_v[k][z][j][i])+" ")
            f.close()
            conv2_weights_v = sess.run(conv2_weights)
            f=open("weight_save/conv2_weights.txt","w")

            for i in range(OUTPUT_CHANNELS2):
                for j in range(OUTPUT_CHANNELS1):
                    for k in range(5):
                        for z in range(5):
                            f.write(str(conv2_weights_v[k][z][j][i])+" ")
            f.close()
     # a=sess.run(conv1_biases)
           # np.savetxt("weight_save/t.txt",a, fmt = "%f")
            np.savetxt("weight_save/conv1_b.txt",sess.run(conv1_biases)  ,fmt="%10.5f")
          #np.savetxt("weight_save/conv2_w.txt",sess.run(conv2_weights) ,fmt="%f")
            np.savetxt("weight_save/conv2_b.txt",sess.run(conv2_biases) ,fmt="%10.5f")   
          
            f=open("weight_save/fc1_w.txt","w")
            fc1_weights_v = sess.run(fc1_weights)
            for i in range(2048):
                for j in range(128):
                    f.write(str(fc1_weights_v[i][j])+" ")
            f.close()
      

      #np.savetxt("weight_save/fc1_w.txt",sess.run(fc1_weights) ,fmt="%f")
            np.savetxt("weight_save/fc1_b.txt",sess.run(fc1_biases) ,fmt="%10.5f")
            f=open("weight_save/fc2_w.txt","w")
            fc2_weights_v = sess.run(fc2_weights)
            for i in range(128):
                for j in range(63):
                    f.write(str(fc2_weights_v[i][j])+" ")
            f.close()

      #np.savetxt("weight_save/fc2_w.txt",sess.run(fc2_weights) ,fmt="%f")
            np.savetxt("weight_save/fc2_b.txt",sess.run(fc2_biases) ,fmt="%10.5f")
      va.close()

    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

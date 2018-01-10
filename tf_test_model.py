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


NUM_CHANNELS =1
OUTPUT_CHANNELS1=16
OUTPUT_CHANNELS2=32
MIDDLE_LAYER = 512
SEED = 66478  # 

############################################
conv1_weights_v = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/weight_save/conv1_weights.txt',dtype=np.float32)

f=open("/Users/YiyueZhang/Desktop/final_project/conv/weight_save/conv1_weightsd.txt","w")
f.write("float conv1_weights[5*5*16] = {")
for i in range(5*5*16):
    f.write(str(conv1_weights_v[i])+",")
f.write("};")
f.close()
#conv1_weights_v = np.reshape(conv1_weights_v,400)
conv1_weights =np.zeros([5,5,1,16],dtype ='float32')
c=0
for i in range(OUTPUT_CHANNELS1):
	for j in range(1):
		for k in range(5):
			for z in range(5):
				conv1_weights[k][z][j][i] = conv1_weights_v[c]
				c = c+1
############################################

conv1_bias = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/weight_save/conv1_b.txt',dtype=np.float32 )

f=open("/Users/YiyueZhang/Desktop/final_project/conv/weight_save/conv1_bd.txt","w")
f.write("float conv1_bias[16] = {")
for i in range(16):
    f.write(str(conv1_bias[i])+",")
f.write("};")
f.close()

############################################
conv2_weights_v = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/weight_save/conv2_weights.txt',dtype=np.float32)
f=open("/Users/YiyueZhang/Desktop/final_project/conv/weight_save/conv2_weightsd.txt","w")
f.write("float conv2_weights[5*5*16*32] = {")
for i in range(5*5*16*32):
    f.write(str(conv2_weights_v[i])+",")
f.write("};")
f.close()
conv2_weights =np.zeros([5,5,16,32],dtype ='float32')
c=0
for i in range(OUTPUT_CHANNELS2):
	for j in range(OUTPUT_CHANNELS1):
		for k in range(5):
			for z in range(5):
				conv2_weights[k][z][j][i] = conv2_weights_v[c]
				c = c+1
#conv2_weights = np.reshape(conv2_weights,[5,5,16,32])

############################################
conv2_bias = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/weight_save/conv2_b.txt',dtype=np.float32 )
f=open("/Users/YiyueZhang/Desktop/final_project/conv/weight_save/conv2_bd.txt","w")
f.write("float conv2_bias[32] = {")
for i in range(32):
    f.write(str(conv2_bias[i])+",")
f.write("};")
f.close()

############################################
                                
fc1_weights_v = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/weight_save/fc1_w.txt',dtype=np.float32)
print(fc1_weights_v.shape)
f=open("/Users/YiyueZhang/Desktop/final_project/conv/weight_save/fc1_wd.txt","w")
f.write("float fc1_weights[2048*128] = {")
for i in range(128*2048):
    f.write(str(fc1_weights_v[i])+",")
f.write("};")
f.close()

fc1_weights = np.zeros([2048,128],dtype ='float32')
c = 0
for i in range(2048):
    for j in range(128):
        fc1_weights[i][j] = fc1_weights_v[c]
        c = c+1

############################################

fc1_bias = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/weight_save/fc1_b.txt',dtype=np.float32 )
f=open("/Users/YiyueZhang/Desktop/final_project/conv/weight_save/fc1_bd.txt","w")
f.write("float fc1_bias[128] = {")
for i in range(128):
    f.write(str(fc1_bias[i])+",")
f.write("};")
f.close()

############################################


fc2_weights_v = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/weight_save/fc2_w.txt',dtype=np.float32)
f=open("/Users/YiyueZhang/Desktop/final_project/conv/weight_save/fc2_wd.txt","w")
f.write("float fc2_weights[63*128] = {")
for i in range(128*63):
    f.write(str(fc2_weights_v[i])+",")
f.write("};")
f.close()
fc2_weights = np.zeros([128,63],dtype ='float32')
c = 0
for i in range(128):
    for j in range(63):
        fc2_weights[i][j] = fc2_weights_v[c]
        c = c+1

############################################
fc2_bias = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/weight_save/fc2_b.txt',dtype=np.float32 )
f=open("/Users/YiyueZhang/Desktop/final_project/conv/weight_save/fc2_bd.txt","w")
f.write("float fc2_bias[63] = {")
for i in range(63):
    f.write(str(fc2_bias[i])+",")
f.write("};")
f.close()
############################################
#display_chars = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/test_data.txt',dtype=np.float32)
'''f=open("/Users/YiyueZhang/Desktop/final_project/conv/test_datad.txt","w")
f.write("{")
for i in range(62):
    f.write("{")
    for j in range(32*32):
        f.write(str(display_chars[i][j])+",")
    f.write("},")
f.write("};")
f.close()'''

############################################



test_data = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/test_data.txt',dtype=np.float32 )
f=open("/Users/YiyueZhang/Desktop/final_project/conv/test_datad.txt","w")
f.write("{")
for i in range(63):
    f.write("{")
    for j in range(32*32):
        f.write(str(test_data[i][j])+",")
    f.write("},")
f.write("};")
f.close()
test_labels = numpy.loadtxt('/Users/YiyueZhang/Desktop/final_project/conv/test_label.txt')

test_data = np.reshape(test_data,[945,32,32,1])
print(test_labels)
############################################

with tf.Session() as sess:
	count = 0;
	for k in range(945):
		input_data = np.reshape(test_data[k],[1,32,32,1])
        #print(input_data.shape)
		conv1 = tf.nn.conv2d(input_data,conv1_weights,strides=[1, 1, 1, 1],padding='SAME')
        #conv1 = sess.run(conv1)
        #print(conv1)
		conl1_r = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
		#conl1_r = sess.run(conl1_r)
		pool1 = tf.nn.max_pool(conl1_r,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
		#pool1 = sess.run(pool1)
		conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1, 1, 1, 1],padding='SAME')
		#conv2 = sess.run(conv2)
		#conv1_r = sess.run(conv1)
		conl2_r = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
		#conl2_r = sess.run(conl2_r)
		pool2 = tf.nn.max_pool(conl2_r,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
		#pool2 - sess.run(pool2)
		pool_shape = pool2.get_shape().as_list()
		reshape = tf.reshape(pool2,[pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

		fc1=tf.matmul(reshape, fc1_weights) + fc1_bias
		#fc1 = sess.run(fc1)
		fc2 = tf.matmul(fc1, fc2_weights) + fc2_bias
		#fc2 = sess.run(fc2)
		test_prediction = tf.nn.softmax(fc2)

		test_prediction = sess.run(test_prediction)
		#print(test_prediction)
		test_prediction = np.reshape(test_prediction, [63])
		#print(test_prediction.shape)
		tmp3 = 0
		for h in range(63):
			if (test_prediction[h] > tmp3):
				result = h;
				tmp3 = test_prediction[h];



		for i in range(63):
			if(test_labels[k][i]==1):
				tmp_result2 = i;
        
		if(tmp_result2 == result):
			count = count +1
		print("result for input %i , predict--%i, actual--%i \n",k, result,tmp_result2);
	print("Accuracy-- %f\n", (count)/(130));
	#print (fc1)
	

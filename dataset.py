import tensorflow as tf
from skimage import data, io, filters
import skimage.transform
import numpy as np 
import random
import timeit
import numpy, scipy.io
import matplotlib.image as mpimg
from scipy.misc import toimage

from PIL import Image
import os

def rgb2gray(rgb):

	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray

def gray2ycbcr(gray):
	ycbcr= np.zeros([1,64*64/2], dtype = 'uint32')
	print(ycbcr.shape)
	j=0
	for i in range(64*32):
		y0 = np.int32(gray[j]/1.16388)
		y1 = np.int32(gray[j+1]/1.16388)
		temp= (y0<<24) | (y1 << 8)
		#print(temp)
		ycbcr[0,i] = temp
		
		j=j+2

	return ycbcr


def load_image_chars74k(image_path_set,display_chars_set,bbox=None):
	clip = np.zeros([54*62+54,32*32], dtype = 'float32')
	'''print(clip.shape)
	k=0
	for k in range(62*55):
		path = '/Users/YiyueZhang/Desktop/final_project/softmax/English/Hnd/' + image_path_set[k]
	
		img = (skimage.io.imread(path))
		img = skimage.transform.resize(img,(32,32))
		img = skimage.img_as_ubyte(img).astype(np.float32)
		img = rgb2gray(img)
		img = (np.reshape(img,32*32))
		bw = np.zeros([256],dtype = 'int')
		i = 0
		for i in range(32*32):
			bw[np.int(img[i])] = bw[np.int(img[i])] +1
		sum_thre =0
		i=0
		thershold = 0
		for i in range(255):
			sum_thre = sum_thre + bw[i];
			if(sum_thre > 32*32*0.2):
				thershold=i;
				break;

		tmp_img = []
		for r in range(32*32):

			if (img[r]>thershold):

				tmp_img.append((1))
				#fh.write("1,")
			else:
				tmp_img.append((0))

		clip [k,:] = tmp_img
	
	c = np.ones([55,32*32],dtype = 'float32')
	k=0
	for k in range(55):
		clip[62*55+k,:] = c[k]'''
	k=0
	for k in range(63):
		path = '/Users/YiyueZhang/Desktop/final_project/softmax/DSP_DATA/' + display_chars_set[k]
		img = (skimage.io.imread(path))
		img = skimage.transform.resize(img,(384/2,576/2))
		img = skimage.img_as_ubyte(img).astype(np.float32)
		img = rgb2gray(img)
		#img = (np.reshape(img,576*384))
		#print(img)
		#for s in range(64):
		#    print(img[s:s+64])
		#with open("text.jpeg","w")as f:
		#    for i,x in enumerate(img) :
		#        if ( i % 720 == 0 and i > 0): f.write("\n")
		#        f.write("%d, "%x )

		#path = '/Users/YiyueZhang/Desktop/final_project/softmax/c.jpeg'
		#img = (skimage.io.imread(path))
		#img = skimage.img_as_ubyte(img).astype(np.uint32)
		loc_x= [0,32,32*2,32*3,32*4,32*5,32*6,32*7,32*8,32*9]
		loc_y= [0,32,32*2,32*3,32*4,32*5,32*6]
		for j in range(6):
			for i in range(9):
				#fh = open("img_%d_%d" % (i,j), "w")
				temp_img= img[loc_y[j]:loc_y[j+1],loc_x[i]:loc_x[i+1]]
				temp_img = (np.reshape(temp_img,32*32))
				#print(temp_img.shape)
				#toimage(temp_img).show()
				
				bw = np.zeros([256],dtype = 'int')
				iq = 0

				for iq in range(32*32):
					bw[np.int(temp_img[iq])] = bw[np.int(temp_img[iq])] +1
				sum_thre =0
				iq=0
				thershold = 0
				for iq in range(255):
					sum_thre = sum_thre + bw[iq];
					if(sum_thre > 32*32*0.2):
						thershold=iq;
						break;
				#print(thershold)
				for ks in range(32*32):
					
					if (temp_img[ks] >thershold):
						temp_img[ks] = 1.0
						#fh.write("1,")
					else:
						temp_img[ks] = 0
						#fh.write("0,")
				
				#print(temp_img)
				#temp_img = (np.reshape(temp_img,(32,32)))
				#toimage(temp_img).show()
				clip [(k*54+j*9+i),:] = temp_img



	print (clip.shape)
	#print(clip)
	return clip

def load_display_chars74k(display_chars_set_lower,display_chars_set,bbox=None):
	clip = np.zeros([62,64*64/2], dtype = 'uint32')

	for k in range(36):
		path = '/Users/YiyueZhang/Desktop/final_project/softmax/display_chars/' + display_chars_set[k]
		img = (skimage.io.imread(path))
		img = skimage.transform.resize(img,(64,64))
		img = skimage.img_as_ubyte(img).astype(np.uint32)
		#print(img)
		#print(img.shape)
		#img = rgb2gray(img)
		img = np.reshape(img,64*64)
		clip [k,:] = gray2ycbcr(img)

	for j in range(26):
		path = '/Users/YiyueZhang/Desktop/final_project/softmax/display_chars/lower/' + display_chars_set_lower[j]
		img = (skimage.io.imread(path))
		img = skimage.transform.resize(img,(64,64))
		img = skimage.img_as_ubyte(img).astype(np.uint32)
		#img = rgb2gray(img)
		img = np.reshape(img,64*64)
		clip [j+36,:] = gray2ycbcr(img)
	return clip



def single_test():

	sample_image_list=[]
	listOfNumbers=[]
	image_test_path_set =[]
	label_of_test_samples=[]
	image_train_path_set =[]
	label_of_train_samples=[]
	image_debug_path_set =[]
	image_debug_label = []

	display_chars_set = ['0.jpeg','1.jpeg','2.jpeg','3.jpeg','4.jpeg','5.jpeg','6.jpeg','7.jpeg','8.jpeg','9.jpeg','A.jpeg','B.jpeg','cs.jpeg','D.jpeg','E.jpeg','F.jpeg','G.jpeg','H.jpeg','I.jpeg','J.jpeg','K.jpeg','L.jpeg','M.jpeg','N.jpeg','O.jpeg','P.jpeg','Q.jpeg','R.jpeg','S.jpeg','T.jpeg','U.jpeg','V.jpeg','W.jpeg','X.jpeg','Y.jpeg','Z.jpeg','as.jpeg','bs.jpeg','cs.jpeg','ds.jpeg','es.jpeg','fs.jpeg','gs.jpeg','hs.jpeg','is.jpeg','js.jpeg','ks.jpeg','ls.jpeg','ms.jpeg','ns.jpeg','os.jpeg','ps.jpeg','qs.jpeg','rs.jpeg','ss.jpeg','ts.jpeg','us.jpeg','vs.jpeg','ws.jpeg','xs.jpeg','ys.jpeg','zs.jpeg','WHITE.jpeg']

	sample_image_list = [line.rstrip('\n') for line in open('/Users/YiyueZhang/Desktop/final_project/softmax/English/Hnd/all.txt~')]

	total = load_image_chars74k(sample_image_list,display_chars_set)


	num_of_trainning_samples = 63*(39)
	num_of_samples = (26+26+10+1)*(54)
	num_of_test_samples = num_of_samples - num_of_trainning_samples

	sample_labels = numpy.zeros((num_of_samples,63))
	k=0



	for j in range(63):
		for i in range (54):
			sample_labels[i+k*54,j] = 1
		k = k +1 
	print(sample_labels)
	

	listOfNumbers= random.sample(range(num_of_samples), num_of_test_samples)


	a = np.zeros([num_of_test_samples,63], dtype = 'int')
	b = np.zeros([num_of_test_samples,32*32], dtype = 'float32')
	#counter = 0
	for y in range(len(listOfNumbers)):
		b[y,:] = (total[listOfNumbers[y],:])
		a[y,:] = (sample_labels[listOfNumbers[y],:])
		#coutner = counter+1
	
	counter = 0
	a_train = np.zeros([num_of_trainning_samples,63], dtype = 'int')
	b_train = np.zeros([num_of_trainning_samples,32*32], dtype = 'float32')
	for z in range(num_of_samples):
		if z in listOfNumbers:
			pass
		else:
			b_train[counter,:] = total[z,:]
			a_train[counter,:] = sample_labels[z,:]
			counter = counter+1

	np.savetxt('test_data.txt', b, fmt="%f")
	np.savetxt('test_label.txt', a, fmt="%d")
	np.savetxt('train_data.txt', b_train, fmt="%f")
	np.savetxt('train_label.txt', a_train, fmt="%d")
'''
	#display_chars_set = ['0.jpeg','1.jpeg','2.jpeg','3.jpeg','4.jpeg','5.jpeg','6.jpeg','7.jpeg','8.jpeg','9.jpeg','A.jpeg','B.jpeg','cs.jpeg','D.jpeg','E.jpeg','F.jpeg','G.jpeg','H.jpeg','I.jpeg','J.jpeg','K.jpeg','L.jpeg','M.jpeg','N.jpeg','O.jpeg','P.jpeg','Q.jpeg','R.jpeg','S.jpeg','T.jpeg','U.jpeg','V.jpeg','W.jpeg','X.jpeg','Y.jpeg','Z.jpeg','as.jpeg','bs.jpeg','cs.jpeg','ds.jpeg','es.jpeg','fs.jpeg','gs.jpeg','hs.jpeg','is.jpeg','js.jpeg','ks.jpeg','ls.jpeg','ms.jpeg','ns.jpeg','os.jpeg','ps.jpeg','qs.jpeg','rs.jpeg','ss.jpeg','ts.jpeg','us.jpeg','vs.jpeg','ws.jpeg','xs.jpeg','ys.jpeg','zs.jpeg','WHITE.jpeg']

	#clip = np.zeros([63,576*384], dtype = 'float32')


	path = '/Users/YiyueZhang/Desktop/final_project/softmax/test_data.txt'
	img = np.loadtxt(path)
	img = np.reshape(img,(945,32,32))
	#img = skimage.transform.resize(img,(576,384))
	#img = skimage.img_as_ubyte(img).astype(np.float32)
	#img = rgb2gray(img)
	toimage(img[20]).show()

'''
'''
	for k in range(63):
		path = '/Users/YiyueZhang/Desktop/final_project/softmax/DSP_DATA/' + display_chars_set[k]
		img = (skimage.io.imread(path))
		img = skimage.transform.resize(img,(576,384))
		img = skimage.img_as_ubyte(img).astype(np.float32)
		img = rgb2gray(img)
		#toimage(img).show()
		
		img = np.reshape(img,576*384)

		bw = np.zeros([256],dtype = 'int')
		iq = 0
		for iq in range(576*384):
			bw[np.int(img[iq])] = bw[np.int(img[iq])] +1
		sum_thre =0
		iq=0
		thershold = 0

		for iq in range(255):
			sum_thre = sum_thre + bw[iq];
			if(sum_thre > 576*384*0.2):
				thershold=iq;
				break;

		for i in range(576*384):
			if(img[i]>thershold):
				img[i] = 1.0
			else:
				img[i] = 0.0

		img_test=np.reshape(img,[3,73728])
		print(img_test[0,:])
		print(img_test[1,:])
		print(img_test[2,:])
		clip[k,:] = img
	np.savetxt('screening.txt', clip, fmt="%f")
'''
	#for k in range(55):
	#	image_debug_path_set.append(sample_image_list[55*21+k])
#	image_debug_label.append(sample_labels[55*21+k])


#test = load_image_chars74k(image_debug_path_set)

#	np.savejpeg('debug_data.jpeg', test, fmt="%f")
#	np.savejpeg('debug_label.jpeg', image_debug_label, fmt="%d")

#	a = load_image_chars74k(image_test_path_set,display_chars_set)
#	b = load_image_chars74k(image_train_path_set,display_chars_set)

	#np.savejpeg('test_data.jpeg', a, fmt="%f")
	#np.savejpeg('test_label.jpeg', label_of_test_samples, fmt="%d")
	#np.savejpeg('train_data.jpeg', b, fmt="%f")
	#np.savejpeg('train_label.jpeg', label_of_train_samples, fmt="%d")
	#display_chars_set = ['0.jpeg','1.jpeg','2.jpeg','3.jpeg','4.jpeg','5.jpeg','6.jpeg','7.jpeg','8.jpeg','9.jpeg','A.jpeg','B.jpeg','C.jpeg','D.jpeg','E.jpeg','F.jpeg','G.jpeg','H.jpeg','I.jpeg','J.jpeg','K.jpeg','L.jpeg','M.jpeg','N.jpeg','O.jpeg','P.jpeg','Q.jpeg','R.jpeg','S.jpeg','T.jpeg','U.jpeg','V.jpeg','W.jpeg','X.jpeg','Y.jpeg','Z.jpeg','as.jpeg','bs.jpeg','cs.jpeg','ds.jpeg','es.jpeg','fs.jpeg','gs.jpeg','hs.jpeg','is.jpeg','js.jpeg','ks.jpeg','ls.jpeg','ms.jpeg','ns.jpeg','os.jpeg','ps.jpeg','qs.jpeg','rs.jpeg','ss.jpeg','ts.jpeg','us.jpeg','vs.jpeg','ws.jpeg','xs.jpeg','ys.jpeg','zs.jpeg','WHITE.jpeg']


	#c = load_display_chars74k(display_chars_set_lower,display_chars_set)
	#np.savejpeg('display_data.jpeg', c, fmt="%d")



if __name__ == "__main__":
	single_test()









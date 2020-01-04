#import os
import sys

#caffe_root = '/home/xy18/workspace/git/caffe/caffe-ssd-20191109/python'
#sys.path.insert(0, caffe_root)
#sys.path.append(caffe_root)

import caffe
import time
import numpy as np
import matplotlib.pyplot as plt

# reference
	# https://blog.csdn.net/Chris_zhangrx/article/details/78744656

	# create net layers
	# http://simtalk.cn/2016/10/28/PyCaffe-in-Practice/


image = 'data/face-640x360.jpg'
#image = 'data/face-1280x720.jpg'

start, end = 0, 0

def track_time(enable):
	if(enable):
		global start
		start = time.clock()
	else:
		global end
		end = time.clock()
		print('[-] Duration:\t{}'.format(end - start))

# https://www.cnblogs.com/denny402/p/5092075.html
def show_dump(data, file):
	# c,h,w -> h,w,c
	data = data.transpose(1,2,0)
	#data -= data.min()
	data /= data.max()
	print('show_dump data shape:\t{}'.format(data.shape))

	plt.imshow(data, cmap='gray')
	plt.savefig(file)
	plt.show()

def inference(file):

	# model
	prototxt = file
	caffemodel = prototxt.split('.')[0] + '.caffemodel'
	print('[-] inference prototxt  :\t{}'.format(prototxt))
	print('[-] inference caffemodel:\t{}'.format(caffemodel))

	# cpu/gpu
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)
	caffe.set_mode_cpu()

	# data
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2, 0, 1)) # 360,640,3 -> 3,360,640
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR

	# grey
	#input_data = caffe.io.load_image(image, color=False)
	# h,w,c
	input_data = caffe.io.load_image(image, color=True)
	plt.imshow(input_data)
	plt.savefig('orig.png')
	plt.show()

	# preprocess: reshape
	net.blobs['data'].data[0] = transformer.preprocess('data', input_data)
	#net.blobs['data'].data[...] = transformer.preprocess('data', input_data)

	# input
	input = net.blobs['data'].data[0]
	show_dump(input, 'input.png')

	# statistics
	track_time(True)
	out = net.forward()
	track_time(False)

	# output
	output = net.blobs['conv1'].data[0]
	show_dump(output, 'output.png')

	loss = net.blobs['loss'].data
	print('loss:\t{}'.format(loss))

def main():
	print('[-] main')

	if len(sys.argv) < 2:
		print('Please specify a model')
		return -1

	inference(sys.argv[1])
	return 0

if __name__ == "__main__":
	sys.exit(main())

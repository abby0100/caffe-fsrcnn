#import os
import sys

caffe_root = '/home/xy18/workspace/git/caffe/caffe-ssd-20191109/python'
sys.path.insert(0, caffe_root)
#sys.path.append(caffe_root)

import caffe
import time
import numpy as np
import matplotlib.pyplot as plt

# reference
	# https://blog.csdn.net/Chris_zhangrx/article/details/78744656

	# create net layers
	# http://simtalk.cn/2016/10/28/PyCaffe-in-Practice/


data_image = 'data/face-640x360.jpg'
label_image = 'data/face-1280x720.jpg'

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
	plt.title(file)
	plt.show()


def inference(file):
	# model
	prototxt = file
	caffemodel = prototxt.split('.')[0] + '.caffemodel'
	print('------------------------------------------------------------')
	print('[-] inference prototxt  :\t{}'.format(prototxt))
	print('[-] inference caffemodel:\t{}'.format(caffemodel))

	# cpu/gpu
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)
	caffe.set_mode_cpu()

	# data
	data_name = 'data'
	data_transformer = caffe.io.Transformer({data_name: net.blobs[data_name].data.shape})
	data_transformer.set_transpose(data_name, (2, 0, 1)) # 360,640,3 -> 3,360,640
	data_transformer.set_raw_scale(data_name, 255)
	data_transformer.set_channel_swap(data_name, (2, 1, 0)) # RGB -> BGR

	# grey
	#data_input = caffe.io.load_image(data_image, color=False)
	# h,w,c
	data_input = caffe.io.load_image(data_image, color=True)
	plt.imshow(data_input)
	data_png = data_name + '.png'
	plt.savefig(data_png)
	plt.title(data_png)
	plt.show()

	net.blobs[data_name].data[0] = data_transformer.preprocess(data_name, data_input)
	#net.blobs[data_name].data[...] = data_transformer.preprocess(data_name, data_input)

	# label
	label_name = 'label'
	label_transformer = caffe.io.Transformer({label_name: net.blobs[label_name].data.shape})
	label_transformer.set_transpose(label_name, (2, 0, 1))
	label_transformer.set_raw_scale(label_name, 255)
	label_transformer.set_channel_swap(label_name, (2, 1, 0))

	label_input = caffe.io.load_image(label_image, color=True)
	plt.imshow(label_input)
	label_png = label_name + '.png'
	plt.savefig(label_png)
	plt.title(label_png)
	plt.show()

	net.blobs[label_name].data[0] = label_transformer.preprocess(label_name, label_input)

	# input
	input = net.blobs[data_name].data[0]
	show_dump(input, 'input.png')

	# summary
	print('------------------------------------------------------------')
	for layer_name, blob in net.blobs.iteritems():
		print(layer_name+'\t'+str(blob.data.shape))

	# statistics
	track_time(True)
	out = net.forward()
	track_time(False)

	# output
	#output = net.blobs['conv1'].data[0]
	output = net.blobs['reshape_to_4D'].data[0]
	show_dump(output, 'output.png')

	loss = net.blobs['loss'].data
	print('------------------------------------------------------------')
	print('loss:\t{}'.format(loss))
	print('total loss:\t{}'.format(loss * 720 * 1280))

def main():
	print('------------------------------------------------------------')
	print('[-] main')

	if len(sys.argv) < 2:
		print('Please specify a model')
		return -1

	inference(sys.argv[1])
	return 0

if __name__ == "__main__":
	sys.exit(main())

# usage:
# python run-caffe-model-20190104.py 640-360/subpixel/fsrcnn-s_deploy.prototxt

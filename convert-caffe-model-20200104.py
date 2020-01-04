import os
import sys
import numpy as np
import caffe


def generate_fsrcnn(prototxt, caffemodel):
	net = caffe.Net(prototxt, caffe.TEST)
	net.save(caffemodel)

def main():
	if len(sys.argv) < 2:
		print('\n[-] Should pass the prototxt to this script, exiting...\n')
		return -1

	file = sys.argv[1]
	prefix = file.split('.')
	print('prototxt:\t{}'.format(file))

	caffemodel = prefix[0] + ".caffemodel"
	print('caffemodel:\t{}'.format(caffemodel))
	
	generate_fsrcnn(file, caffemodel)
	return 0

if __name__ == "__main__":
	sys.exit(main())

# python convert-caffe-model-20200104.py test-io/fsrcnn-s_deploy.prototxt

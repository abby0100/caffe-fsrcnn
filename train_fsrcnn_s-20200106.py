# https://www.cnblogs.com/denny402/p/5684431.html
import caffe
from caffe import layers as L, params as P, proto, to_proto

root = '/root/e3s10/caffe-ssd/examples/xy_fsrcnn_s/640-360/subpixel/'
train_list = root + 'train.txt'
train_label_list = root + 'train_label.txt'
test_list  = root + 'test.txt'
test_label_list = root + 'test_label.txt'
train_proto= root + 'train.prototxt'
test_proto = root + 'test.prototxt'

train_test_proto = root + 'fsrcnn-s_train_test.prototxt'
solver_proto = root + 'solver.prototxt'

#def FSRCNN_s(img_list, batch_size, include_acc = False):
def FSRCNN_s(img_list, label_list, batch_size, include_acc = False):
	print('Create FSRCNN_s')
	# data
	# https://www.cnblogs.com/houjun/p/9909764.html
	#data, label = L.ImageData(
	data = L.ImageData(
			name="data",
			ntop=2,
			#include={'phase': caffe.TRAIN})
			source=img_list,
			batch_size=batch_size,
			is_color=True,
			new_width=640,
			new_height=360,
			#shuffle=True,
			root_folder=root,
			transform_param=dict(
				#crop_size=360,
				scale = 0.00390625,
				#mirror=True
				))
	# label
	label = L.ImageData(
			name="label",
			ntop=2,
			source=label_list,
			batch_size=batch_size,
			is_color=True,
			new_width=1280,
			new_height=720,
			#shuffle=True,
			root_folder=root,
			transform_param=dict(
				#crop_size=720,
				scale = 0.00390625,
				#mirror=True
				))
	# https://www.cnblogs.com/houjun/p/9909764.html
	#label = L.HDF5Data(
	#		name="label",
	#		ntop=2,
	#		source=img_list,
	#		#source=label_list,
	#		batch_size=batch_size,
	#		include=dict(phase=caffe.TRAIN))
	#label = L.HDF5Data(
	#		hdf5_data_param={
	#			'source': img_list,
	#			'batch_size': 64},
	#		include={
	#			'phase': caffe.TRAIN})

	# conv1
	conv1 = L.Convolution(
			data,
			#label,
			name="conv1",
			num_output=32,
			kernel_size=5,
			stride=1,
			pad=1,
			weight_filler=dict(type='gaussian', std=0.05),
			bias_filler=dict(type='constant', value=0))
	relu1 = L.PReLU(
			conv1,
			name="relu1",
			in_place=True,
			prelu_param={
				'channel_shared': 1})
	# conv2
	conv2 = L.Convolution(
			conv1,
			name="conv2",
			num_output=5,
			kernel_size=1,
			stride=1,
			pad=0,
			group=1,
			weight_filler=dict(type='gaussian', std=0.05),
			bias_filler=dict(type='constant', value=0))
	relu2 = L.PReLU(
			conv2,
			name="relu2",
			in_place=True,
			prelu_param={
				'channel_shared': 1})
	# conv22
	conv22 = L.Convolution(
			conv2,
			name="conv22",
			num_output=5,
			kernel_size=3,
			stride=1,
			pad=1,
			group=1,
			weight_filler=dict(type='gaussian', std=0.05),
			bias_filler=dict(type='constant', value=0))
	relu22 = L.PReLU(
			conv22,
			name="relu22",
			in_place=True,
			prelu_param={
				'channel_shared': 1})
	# conv23
	conv23 = L.Convolution(
			conv22,
			name="conv23",
			num_output=32,
			kernel_size=1,
			stride=1,
			pad=1,
			group=1,
			weight_filler=dict(type='gaussian', std=0.05),
			bias_filler=dict(type='constant', value=0))
	relu23 = L.PReLU(
			conv23,
			name="relu23",
			in_place=True,
			prelu_param={
				'channel_shared': 1})
	# conv3
	conv3 = L.Convolution(
			conv23,
			name="conv3",
			num_output=12,
			kernel_size=3,
			stride=1,
			pad=1,
			weight_filler=dict(type='gaussian', std=0.05),
			bias_filler=dict(type='constant', value=0))
	# shuffle
	reshape1 = L.Reshape(
			conv3,
			name="reshape_to_6d",
			shape={
			#reshape_param={
			#	'shape'={
					'dim': 0,
					'dim': 2,
					'dim': 2,
					'dim': 3,
					'dim': 360,
					'dim': -1}
			#	})
			)
	permute = L.Permute(
			reshape1,
			name="permute",
			permute_param={
				'order': 0,
				'order': 3,
				'order': 4,
				'order': 1,
				'order': 5,
				'order': 2})
	reshape2 = L.Reshape(
			permute,
			name="reshape_to_4d",
			shape={
				'dim': 0,
				'dim': 3,
				'dim': 720,
				'dim': -1}
			)
	# loss
	loss = L.EuclideanLoss(
			reshape2,
			label,
			name="loss")
	
	#return to_proto(conv1)
	#return to_proto(label, conv1)
	#return to_proto(data, label, conv1, relu1)
	#return to_proto(data, label, relu1)
	#return to_proto(data, label, relu1, relu2, relu22, relu23, conv3, reshape1)
	#return to_proto(data, label, relu1, relu2, relu22, relu23, conv3, loss)
	return to_proto(data, label, relu1, relu2, relu22, relu23, conv3, reshape2, loss)

##################################################################################################
# http://wentaoma.com/2016/08/10/caffe-python-common-api-reference/
def generate_net():
	# write train.prototxt
	with open(train_proto, 'w') as f:
		f.write(str(FSRCNN_s(train_list, train_label_list, batch_size=1)))

	# write test.prototxt
	with open(test_proto, 'w') as f:
		f.write(str(FSRCNN_s(test_list, test_label_list, batch_size=1, include_acc=True)))

def train_from_solver():
	print('train_from_solver')
	#solver = caffe.SGDSolver(solver_proto)
	solver = caffe.get_solver(solver_proto)

	#solver.net.forward()
	#solver.test_nets[0].forward()
	#solver.net.backward()
	#solver.step(1)
	print('[-] start to train...')
	solver.solve()
	print('[-] start to train...')

	#solver.net.save('fsrcnn-s.caffemodel')

##################################################################################################
def show_usage():
	print('-------------------------------------------------------------------------------------')
	print('[*] 1	generate_net')
	print('[*] 2	train_from_solver')
	chooice = int(input('Pleash chooice action [num] '))
	print('-------------------------------------------------------------------------------------')

	if chooice == 1:
		generate_net()
	elif chooice == 2:
		train_from_solver()

if __name__ == "__main__":
	show_usage()

# usage:
# python train_fsrcnn_s-20200106.py [read 1]

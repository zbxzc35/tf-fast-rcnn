from roi_data.roidata import RoIData
import tensorflow as tf
import pdb

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def build_net(roidb):
	
	name_dict = {}

	roidata = RoIData()
	roidata.set_roidb(roidb)
	
	data = tf.placeholder(tf.float32)
	rois = tf.placeholder(tf.float32)
	labels = tf.placeholder(tf.float32)
	bbox_targets = tf.placeholder(tf.float32)
	bbox_inside_weights = tf.placeholder(tf.float32)
	bbox_outside_weights = tf.placeholder(tf.float32)
	top = [data, rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]
	roidata.setup(top)
	
	name_dict["data"] = data.name
	name_dict["rois"] = rois.name
	name_dict["labels"] = labels.name
	name_dict["bbox_targets"] = bbox_targets.name
	name_dict["bbox_inside_weights"] = bbox_inside_weights.name
	name_dict["bbox_outside_weights"] = bbox_outside_weights.name
	
	W_conv1 = weight_variable([7, 7, 3, 96])
	b_conv1 = bias_variable([96])
	h_conv1 = tf.nn.conv2d(data, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
	h_relu1 = tf.nn.relu(h_conv1)
	h_norm1 = tf.nn.local_response_normalization(h_relu1, depth_radius=3, alpha=0.00005, beta=0.75)
	h_pool1 = tf.nn.max_pool(h_norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	W_conv2 = weight_variable([5, 5, 96, 256])
	b_conv2 = bias_variable([256])
	h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
	h_relu2 = tf.nn.relu(h_conv2)
	h_norm2 = tf.nn.local_response_normalization(h_relu2, depth_radius=3, alpha=0.00005, beta=0.75)
	h_pool2 = tf.nn.max_pool(h_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	W_conv3 = weight_variable([3, 3, 256, 384])
	b_conv3 = bias_variable([384])
	h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3
	h_relu3 = tf.nn.relu(h_conv3)
	
	W_conv4 = weight_variable([3, 3, 384, 384])
	b_conv4 = bias_variable([384])
	h_conv4 = tf.nn.conv2d(h_relu3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4
	h_relu4 = tf.nn.relu(h_conv4)
	
	W_conv5 = weight_variable([3, 3, 384, 256])
	b_conv5 = bias_variable([256])
	h_conv5 = tf.nn.conv2d(h_relu4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5
	h_relu5 = tf.nn.relu(h_conv5)
	
	name_dict["h_relu5"] = h_relu5.name
	
	
	###########################################################################
	
	pooled_height = 6
	pooled_width = 6
	spatial_scale = 0.0625
	
	channels = int(h_relu5.get_shape()[3])
	height = int(h_relu5.get_shape()[1])
	width = int(h_relu5.get_shape()[2])
	
	roi_pool_conv5 = tf.placeholder(tf.float32)
	roi_pool_conv5.set_shape((h_relu5.get_shape()[0], 6, 6, 3))
	
#	num_rois = rois.get_shape()[0]
	num_rois = tf.constant([1, 2, 3]).eval()[2]
	batch_size = h_relu5.get_shape()[0]
	top_count = h_relu5.get_shape()
	'''
	for i in range(num_rois):
		
		roi_batch_ind = rois[i, 0]
		roi_start_w = tf.round(tf.mul(rois[i, 1], spatial_scale))
		roi_start_h = tf.round(tf.mul(rois[i, 2], spatial_scale))
		roi_end_w = tf.round(tf.mul(rois[i, 3], spatial_scale))
		roi_end_h = tf.round(tf.mul(rois[i, 4], spatial_scale))
		
		roi_height = tf.maximum(tf.add(tf.sub(roi_end_h, roi_start_h), 1), 1)
		roi_width = tf.maximum(tf.add(tf.sub(roi_end_w, roi_start_w), 1), 1)
		bin_size_h = tf.div(roi_height, pooled_height)
		bin_size_w = tf.div(roi_width, pooled_width)
		
#		roi_pool_conv5 = tf.nn.max_pool(h_relu5, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')
		
		for c in range(channels):
			for ph in range(pooled_height):
				for pw in range(pooled_width):
					
					hstart = tf.floor(tf.mul(float(ph), bin_size_h))
					wstart = tf.floor(tf.mul(float(pw), bin_size_w))
					hend = tf.ceil(tf.mul(float(ph + 1), bin_size_h))
					wend = tf.ceil(tf.mul(float(pw + 1), bin_size_w))
					
					hstart = tf.minimum(tf.maximum(tf.add(hstart, roi_start_h), 0), height)
					hend = tf.minimum(tf.maximum(tf.add(hend, roi_start_h), 0), height)
					wstart = tf.minimum(tf.maximum(tf.add(wstart, roi_start_w), 0), width)
					wend = tf.minimum(tf.maximum(tf.add(wend, roi_start_w), 0), width)
					
					pool_index = tf.add(tf.mul(ph, pooled_width), pw)
					
					pdb.set_trace()
					
					for h in range(hstart.eval(), hend.eval()):
						for w in range(wstart.eval(), wend.eval()):
							index = tf.add(tf.mul(h, width), w)
							if (batch_data[index] > roi_pool_conv5[pool_index]):
								top_data[pool_index] = batch_data[index]
	
	W_fc6 = weight_variable([6 * 6 * roi_pool_conv5.eval(), 4096])
	b_fc6 = bias_variable([4096])
	h_fc6 = tf.matmul(roi_pool_conv5, W_fc6) + b_fc6
	h_relu6 = tf.nn.relu(h_fc6)
	h_drop6 = tf.nn.dropout(h_relu6, keep_prob=0.5)
	
	W_fc7 = weight_variable([4096, 4096])
	b_fc7 = bias_variable([4096])
	h_fc7 = tf.matmul(h_drop6, W_fc7) + b_fc7
	h_relu7 = tf.nn.relu(h_fc7)
	h_drop7 = tf.nn.dropout(h_relu7, keep_prob=0.5)
	
	W_cls_score = weight_variable([4096, 21])
	b_cls_score = bias_variable([21])
	h_cls_score = tf.matmul(h_drop7, W_cls_score) + b_cls_score
	
	pdb.set_trace()
	
	'''
	
	return roidata, name_dict

def train_step(sess, roidata, name_dict):
	g = tf.get_default_graph()
	
	data = g.get_tensor_by_name(name_dict["data"])
	rois = g.get_tensor_by_name(name_dict["rois"])
	labels = g.get_tensor_by_name(name_dict["labels"])
	bbox_targets = g.get_tensor_by_name(name_dict["bbox_targets"])
	bbox_inside_weights = g.get_tensor_by_name(name_dict["bbox_inside_weights"])
	bbox_outside_weights = g.get_tensor_by_name(name_dict["bbox_outside_weights"])
	top = [data, rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]
	
	feeder, feeder_dict = roidata.forward(top)
	
	feeder_dict_fine = {}
	for i in range(6):
		feeder_dict_fine[top[i]] = feeder[feeder_dict[i]]
	
	sess.run(tf.initialize_all_variables())
	
	h_relu5 = g.get_tensor_by_name(name_dict["h_relu5"])
	result = h_relu5.eval(feed_dict=feeder_dict_fine)
	
	
	pdb.set_trace()



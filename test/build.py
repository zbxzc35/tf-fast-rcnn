import tensorflow as tf
import numpy as np
import pdb

def build_cnn(weight_list, bias_list, name_dict_params):
	sess = tf.InteractiveSession()
	name_dict = {}
	
	data = tf.placeholder(tf.float32)
	
	name_dict["data"] = data.name
	
	### conv1, relu1, norm1 and pool1 ###
	
	weight = np.array(weight_list[name_dict_params["conv1"]])
	w_conv1 = tf.Variable(tf.constant(change_shape(weight).astype(np.float32, copy=False)))
	bias = np.array(bias_list[name_dict_params["conv1"]])
	b_conv1 = tf.Variable(tf.constant(bias.astype(np.float32, copy=False)))
	
	h_conv1 = tf.nn.conv2d(data, w_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
	h_relu1 = tf.nn.relu(h_conv1)
	h_norm1 = tf.nn.local_response_normalization(h_relu1, depth_radius=3, alpha=0.00005, beta=0.75)
	h_pool1 = tf.nn.max_pool(h_norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	### conv2, relu2, norm2 and pool2 ###
	
	weight = np.array(weight_list[name_dict_params["conv2"]])
	w_conv2 = tf.Variable(tf.constant(change_shape(weight).astype(np.float32, copy=False)))
	bias = np.array(bias_list[name_dict_params["conv2"]])
	b_conv2 = tf.Variable(tf.constant(bias.astype(np.float32, copy=False)))
	
	h_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
	h_relu2 = tf.nn.relu(h_conv2)
	h_norm2 = tf.nn.local_response_normalization(h_relu2, depth_radius=3, alpha=0.00005, beta=0.75)
	h_pool2 = tf.nn.max_pool(h_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	### conv3 and relu3 ###
	
	weight = np.array(weight_list[name_dict_params["conv3"]])
	w_conv3 = tf.Variable(tf.constant(change_shape(weight).astype(np.float32, copy=False)))
	bias = np.array(bias_list[name_dict_params["conv3"]])
	b_conv3 = tf.Variable(tf.constant(bias.astype(np.float32, copy=False)))
	
	h_conv3 = tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3
	h_relu3 = tf.nn.relu(h_conv3)
	
	### conv4 and relu4 ###
	
	weight = np.array(weight_list[name_dict_params["conv4"]])
	w_conv4 = tf.Variable(tf.constant(change_shape(weight).astype(np.float32, copy=False)))
	bias = np.array(bias_list[name_dict_params["conv4"]])
	b_conv4 = tf.Variable(tf.constant(bias.astype(np.float32, copy=False)))
	
	h_conv4 = tf.nn.conv2d(h_relu3, w_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4
	h_relu4 = tf.nn.relu(h_conv4)
	
	### conv5 and relu5 ###
	
	weight = np.array(weight_list[name_dict_params["conv5"]])
	w_conv5 = tf.Variable(tf.constant(change_shape(weight).astype(np.float32, copy=False)))
	bias = np.array(bias_list[name_dict_params["conv5"]])
	b_conv5 = tf.Variable(tf.constant(bias.astype(np.float32, copy=False)))
	
	h_conv5 = tf.nn.conv2d(h_relu4, w_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5
	h_relu5 = tf.nn.relu(h_conv5)
	
	name_dict["h_relu5"] = h_relu5.name
	
	sess.run(tf.initialize_all_variables())
	
	return sess, tf.get_default_graph(), name_dict

def build_mlp(weight_list, bias_list, name_dict_params):
	sess = tf.InteractiveSession()
	name_dict = {}
	
	roi_pool_conv5 = tf.placeholder(tf.float32)
	name_dict = {}
	
	name_dict["roi_pool_conv5"] = roi_pool_conv5.name
	
	### fc6, relu6 and drop 6 ###
	
	weight = np.array(weight_list[name_dict_params["fc6"]])
	w_fc6 = tf.Variable(tf.constant(np.swapaxes(weight, 0, 1).astype(np.float32, copy=False)))
	bias = np.array(bias_list[name_dict_params["fc6"]])
	b_fc6 = tf.Variable(tf.constant(bias.astype(np.float32, copy=False)))
	
	h_fc6 = tf.matmul(roi_pool_conv5, w_fc6) + b_fc6
	h_relu6 = tf.nn.relu(h_fc6)
	h_drop6 = tf.nn.dropout(h_relu6, keep_prob=0.5)
	
	### fc7, relu7 and drop 7 ###
	
	weight = np.array(weight_list[name_dict_params["fc7"]])
	w_fc7 = tf.Variable(tf.constant(np.swapaxes(weight, 0, 1).astype(np.float32, copy=False)))
	bias = np.array(bias_list[name_dict_params["fc7"]])
	b_fc7 = tf.Variable(tf.constant(bias.astype(np.float32, copy=False)))
	
	h_fc7 = tf.matmul(h_drop6, w_fc7) + b_fc7
	h_relu7 = tf.nn.relu(h_fc7)
	h_drop7 = tf.nn.dropout(h_relu7, keep_prob=0.5)
	
	### cls_score ###
	
	weight = np.array(weight_list[name_dict_params["cls_score"]])
	w_cls_score = tf.Variable(tf.constant(np.swapaxes(weight, 0, 1).astype(np.float32, copy=False)))
	bias = np.array(bias_list[name_dict_params["cls_score"]])
	b_cls_score = tf.Variable(tf.constant(bias.astype(np.float32, copy=False)))
	
	h_cls_score = tf.matmul(h_drop7, w_cls_score) + b_cls_score
	
	name_dict["h_cls_score"] = h_cls_score.name
	
	### bbox_pred ###
	
	weight = np.array(weight_list[name_dict_params["bbox_pred"]])
	w_bbox_pred = tf.Variable(tf.constant(np.swapaxes(weight, 0, 1).astype(np.float32, copy=False)))
	bias = np.array(bias_list[name_dict_params["bbox_pred"]])
	b_bbox_pred = tf.Variable(tf.constant(bias.astype(np.float32, copy=False)))
	
	h_bbox_pred = tf.matmul(h_drop7, w_bbox_pred) + b_bbox_pred
	
	name_dict["h_bbox_pred"] = h_bbox_pred.name
	
	### cls_prob ###
	
	h_cls_prob = tf.nn.softmax(h_cls_score)
	
	name_dict["h_cls_prob"] = h_cls_prob.name
	
	sess.run(tf.initialize_all_variables())
	
	return sess, tf.get_default_graph(), name_dict

def change_shape(matrix):
	matrix = np.swapaxes(matrix, 0, 1)
	matrix = np.swapaxes(matrix, 1, 2)
	matrix = np.swapaxes(matrix, 2, 3)
	matrix = np.swapaxes(matrix, 0, 1)
	matrix = np.swapaxes(matrix, 1, 2)
	
	return matrix

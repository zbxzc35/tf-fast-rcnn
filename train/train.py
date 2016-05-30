import caffe
from fast_rcnn.config import cfg
import roi_data.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2

import tensorflow as tf
from build_train_test import build_net
from build_train_test import train_step

class SolverWrapper(object):

	def __init__(self, sess, roidb, output_dir):
		self.output_dir = output_dir
		
		if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
			assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED
		
		if cfg.TRAIN.BBOX_REG:
			print 'Computing bounding-box regression targets...'
			self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
			print 'done'
		
		self.roidata, self.name_dict = build_net(roidb)

	def snapshot(self):
		net = self.solver.net
		
		scale_bbox_params = (cfg.TRAIN.BBOX_REG and cfg.TRAIN.BBOX_NORMALIZE_TARGETS and net.params.has_key('bbox_pred'))
		
		if scale_bbox_params:
			orig_0 = net.params['bbox_pred'][0].data.copy()
			orig_1 = net.params['bbox_pred'][1].data.copy()
			
			net.params['bbox_pred'][0].data[...] = (net.params['bbox_pred'][0].data * self.bbox_stds[:, np.newaxis])
			net.params['bbox_pred'][1].data[...] = (net.params['bbox_pred'][1].data * self.bbox_stds + self.bbox_means)
		
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
		
		infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
			if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
		filename = (self.solver_param.snapshot_prefix + infix + '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
		filename = os.path.join(self.output_dir, filename)
		
		net.save(str(filename))
		print 'Wrote snapshot to: {:s}'.format(filename)
		
		if scale_bbox_params:
			net.params['bbox_pred'][0].data[...] = orig_0
			net.params['bbox_pred'][1].data[...] = orig_1
		return filename
	
	def train_model(self, sess, max_iters):
		last_snapshot_iter = -1
		timer = Timer()
		model_paths = []
		for i in range(max_iters):
			timer.tic()
			train_step(sess, self.roidata, self.name_dict)
			timer.toc()
			
			if i % 1000 == 0:
				print 'speed: {:.3f}s / iter'.format(timer.average_time)
			
			if i % cfg.TRAIN.SNAPSHOT_ITERS == 0:
				last_snapshot_iter = i
				model_paths.append(self.snapshot())
		
		if last_snapshot_iter != i:
			model_paths.append(self.snapshot())
		
		return model_paths

def get_training_roidb(imdb):
	if cfg.TRAIN.USE_FLIPPED:
		print 'Appending horizontally-flipped training examples...'
		imdb.append_flipped_images()
		print 'done'
	
	print 'Preparing training data...'
	rdl_roidb.prepare_roidb(imdb)
	print 'done'
	
	return imdb.roidb

def train_net(sess, roidb, output_dir, max_iters=400):
	sw = SolverWrapper(sess, roidb, output_dir)
	
	print 'Solving...'
	model_paths = sw.train_model(sess, max_iters)
	print 'done solving'
	return model_paths

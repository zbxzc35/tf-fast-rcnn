import sys
sys.path.append("../tools")
import _init_paths
from train import get_training_roidb, train_net
from fast_rcnn.config import cfg, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np

import tensorflow as tf
import pdb

gpu_id = 0
max_iters = 400
imdb_name = "voc_2007_trainval"

def combined_roidb(imdb_names):
	def get_roidb(imdb_name):
		imdb = get_imdb(imdb_name)
		print 'Loaded dataset `{:s}` for training'.format(imdb.name)
		imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
		print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
		roidb = get_training_roidb(imdb)
		return roidb
	
	roidbs = [get_roidb(s) for s in imdb_names.split('+')]
	roidb = roidbs[0]
	if len(roidbs) > 1:
		for r in roidbs[1:]:
			roidb.extend(r)
		imdb = datasets.imdb(imdb_names)
	else:
		imdb = get_imdb(imdb_names)
	return imdb, roidb

if __name__ == '__main__':
	cfg.GPU_ID = gpu_id
	
	print('Using config:')
	pprint.pprint(cfg)
	
	np.random.seed(cfg.RNG_SEED)
	caffe.set_random_seed(cfg.RNG_SEED)
	
	#caffe.set_mode_gpu()
	#caffe.set_device(gpu_id)
	import tensorflow as tf
	sess = tf.InteractiveSession()
	
	imdb, roidb = combined_roidb(imdb_name)
	print '{:d} roidb entries'.format(len(roidb))
	
	output_dir = get_output_dir(imdb, None)
	print 'Output will be saved to `{:s}`'.format(output_dir)
	
	train_net(sess, roidb, output_dir, max_iters)

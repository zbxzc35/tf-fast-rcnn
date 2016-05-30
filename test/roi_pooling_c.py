import tensorflow as tf
import numpy as np
from math import floor, ceil
import pdb

from multiprocessing import Pool
from functools import partial

from roi_pool_c import roi_pool_c

def roi_pool(feature_maps, input_rois):
	num_rois = input_rois.shape[0]
	
	pool = Pool()
	func = partial(roi_pool_c, feature_maps)
	roi_pool_conv5s = pool.map(func, input_rois)
	pool.close()
	pool.join()
	
	return roi_pool_conv5s

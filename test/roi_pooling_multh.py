import tensorflow as tf
import numpy as np
from math import floor, ceil
import pdb

from multiprocessing import Pool
from functools import partial

def roi_pool(feature_maps, input_rois):
	num_rois = input_rois.shape[0]
	
	pool = Pool()
	func = partial(roi_pool_sing, feature_maps)
	roi_pool_conv5s = pool.map(func, input_rois)
	pool.close()
	pool.join()
	
	return roi_pool_conv5s

def roi_pool_sing(feature_maps, input_roi):
	roi_pool_conv5 = np.zeros([256, 36], dtype=np.float32)
	
	pooled_height = 6
	pooled_width = 6
	spatial_scale = 0.0625
	
	batch_size, height, width, channels = feature_maps.shape
	
	roi_batch_ind = input_roi[0]
	roi_start_w = round(input_roi[1] * spatial_scale)
	roi_start_h = round(input_roi[2] * spatial_scale)
	roi_end_w = round(input_roi[3] * spatial_scale)
	roi_end_h = round(input_roi[4] * spatial_scale)
	
	roi_height = max(roi_end_h - roi_start_h + 1, 1)
	roi_width = max(roi_end_w - roi_start_w + 1, 1)
	bin_size_h = roi_height / float(pooled_height)
	bin_size_w = roi_width / float(pooled_width)
	
	for c in xrange(channels):
		for ph in xrange(pooled_height):
			for pw in xrange(pooled_width):
				hstart = floor(ph * bin_size_h)
				wstart = floor(pw * bin_size_w)
				hend = ceil((ph + 1) * bin_size_h)
				wend = ceil((pw + 1) * bin_size_w)
				
				hstart = min(max(hstart + roi_start_h, 0), height)
				hend = min(max(hend + roi_start_h, 0), height)
				wstart = min(max(wstart + roi_start_w, 0), width)
				wend = min(max(wend + roi_start_w, 0), width)
				
				pool_index = ph * pooled_width + pw
				
				'''
				is_empty = (hend <= hstart) or (wend <= wstart)
				if (is_empty):
					roi_pool_conv5[c, pool_index] = 0
				'''
				
				for h in xrange(int(hstart), int(hend)):
					for w in xrange(int(wstart), int(wend)):
#						index = h * width + w
						if (feature_maps[0, h, w, c] > roi_pool_conv5[c, pool_index]):
							roi_pool_conv5[c, pool_index] = feature_maps[0, h, w, c]
	
	return roi_pool_conv5

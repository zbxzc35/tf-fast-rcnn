import tensorflow as tf
import numpy as np
import pdb

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.cumath as cumath

def roi_pool(feature_maps, input_rois):
	feature_maps_gpu = gpuarray.to_gpu(feature_maps)
	input_rois_gpu = gpuarray.to_gpu(input_rois)
	
	roi_pool_conv5 = gpuarray.to_gpu(np.zeros([256, 36], dtype=np.float32))
	
	pooled_height = 6
	pooled_width = 6
	spatial_scale = 0.0625
	
	batch_size, height, width, channels = feature_maps_gpu.shape
	num_rois = input_rois_gpu.shape[0]
	
	roi_pool_conv5s = gpuarray.to_gpu(np.zeros([num_rois, 9216], dtype=np.float32))
	for i in range(num_rois):
#		roi_batch_ind = input_rois_gpu[i, 0]
		roi_start_w = cumath.floor(input_rois_gpu[i, 1] * spatial_scale) # should be round()
		roi_start_h = cumath.floor(input_rois_gpu[i, 2] * spatial_scale) # should be round()
		roi_end_w = cumath.floor(input_rois_gpu[i, 3] * spatial_scale) # should be round()
		roi_end_h = cumath.floor(input_rois_gpu[i, 4] * spatial_scale) # should be round()
		
		roi_height = gpuarray.maximum(roi_end_h - roi_start_h + 1, 1)
		roi_width = gpuarray.maximum(roi_end_w - roi_start_w + 1, 1)
		bin_size_h = roi_height / float(pooled_height)
		bin_size_w = roi_width / float(pooled_width)
		
		for c in range(channels):
			for ph in range(pooled_height):
				for pw in range(pooled_width):
					hstart = cumath.floor(ph * bin_size_h)
					wstart = cumath.floor(pw * bin_size_w)
					hend = cumath.ceil((ph + 1) * bin_size_h)
					wend = cumath.ceil((pw + 1) * bin_size_w)
					
					hstart = gpuarray.minimum(gpuarray.maximum(hstart + roi_start_h, 0), height)
					hend = gpuarray.minimum(gpuarray.maximum(hend + roi_start_h, 0), height)
					wstart = gpuarray.minimum(gpuarray.maximum(wstart + roi_start_w, 0), width)
					wend = gpuarray.minimum(gpuarray.maximum(wend + roi_start_w, 0), width)
					
					is_empty = (hend <= hstart) + (wend <= wstart)
					
					pool_index = ph * pooled_width + pw
					if (is_empty.get()):
						roi_pool_conv5[c, pool_index] = 0
					
					for h in range(int(hstart.get()), int(hend.get())):
						for w in range(int(wstart.get()), int(wend.get())):
#							index = h * width + w
							if ((feature_maps_gpu[0, h, w, c] > roi_pool_conv5[c, pool_index]).get()):
								roi_pool_conv5[c, pool_index] = feature_maps_gpu[0, h, w, c]
		
		roi_pool_conv5s[i] = roi_pool_conv5.reshape([9216])
	
	return roi_pool_conv5s
	

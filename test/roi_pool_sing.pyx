import cython
import numpy as np
cimport numpy as np

cdef struct rpc:
	float data[256][36]

cdef extern rpc roi_pool(float *feature_maps, float *input_roi)

@cython.boundscheck(False)
@cython.wraparound(False)
def roi_pool_c(np.ndarray[float, ndim=4, mode="c"] feature_maps, np.ndarray[float, ndim=1, mode="c"] input_roi):
	roi_pool_conv5 =  roi_pool(<float*> feature_maps.data, <float*> input_roi.data)
	
	result = np.zeros([256, 36], dtype=np.float32)
	for i in xrange(256):
		for j in xrange(36):
			result[i, j] = roi_pool_conv5.data[i][j]
	
	return result

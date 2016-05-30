from fast_rcnn.config import cfg
from minibatch import get_minibatch
import numpy as np
from multiprocessing import Process, Queue

import pdb

class RoIData(object):
	
	def __init__(self):
		pass
	
	def _shuffle_roidb_inds(self):
		if cfg.TRAIN.ASPECT_GROUPING:
			widths = np.array([r['width'] for r in self._roidb])
			heights = np.array([r['height'] for r in self._roidb])
			horz = (widths >= heights)
			vert = np.logical_not(horz)
			horz_inds = np.where(horz)[0]
			vert_inds = np.where(vert)[0]
			inds = np.hstack((
				np.random.permutation(horz_inds),
				np.random.permutation(vert_inds)))
			inds = np.reshape(inds, (-1, 2))
			row_perm = np.random.permutation(np.arange(inds.shape[0]))
			inds = np.reshape(inds[row_perm, :], (-1,))
			self._perm = inds
		else:
			self._perm = np.random.permutation(np.arange(len(self._roidb)))
		self._cur = 0

	def _get_next_minibatch_inds(self):
		if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
			self._shuffle_roidb_inds()

		db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
		self._cur += cfg.TRAIN.IMS_PER_BATCH
		return db_inds

	def _get_next_minibatch(self):
		if cfg.TRAIN.USE_PREFETCH:
			return self._blob_queue.get()
		else:
			db_inds = self._get_next_minibatch_inds()
			minibatch_db = [self._roidb[i] for i in db_inds]
			return get_minibatch(minibatch_db, self._num_classes)

	def set_roidb(self, roidb):
		self._roidb = roidb
		self._shuffle_roidb_inds()
		if cfg.TRAIN.USE_PREFETCH:
			self._blob_queue = Queue(10)
			self._prefetch_process = BlobFetcher(self._blob_queue,
												 self._roidb,
												 self._num_classes)
			self._prefetch_process.start()
			def cleanup():
				print 'Terminating BlobFetcher'
				self._prefetch_process.terminate()
				self._prefetch_process.join()
			import atexit
			atexit.register(cleanup)

	def setup(self, top):
		
#		# parse the layer parameter string, which must be valid YAML
#		layer_params = yaml.load(self.param_str_)
#		self._num_classes = layer_params['num_classes']
		self._num_classes = 21
		
		self._name_to_top_map = {}
		
		idx = 0
#		top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
		top[idx].set_shape((cfg.TRAIN.IMS_PER_BATCH, None, None, 3))
		self._name_to_top_map['data'] = idx
		idx += 1

		if cfg.TRAIN.HAS_RPN:
			top[idx].reshape(1, 3)
			self._name_to_top_map['im_info'] = idx
			idx += 1

			top[idx].reshape(1, 4)
			self._name_to_top_map['gt_boxes'] = idx
			idx += 1
		else:
#			top[idx].reshape(1, 5)
			top[idx].set_shape((None, 5))
			self._name_to_top_map['rois'] = idx
			idx += 1

#			top[idx].reshape(1)
			top[idx].set_shape(None)
			self._name_to_top_map['labels'] = idx
			idx += 1

			if cfg.TRAIN.BBOX_REG:
#				top[idx].reshape(1, self._num_classes * 4)
				top[idx].set_shape((None, self._num_classes * 4))
				self._name_to_top_map['bbox_targets'] = idx
				idx += 1

#				top[idx].reshape(1, self._num_classes * 4)
				top[idx].set_shape((None, self._num_classes * 4))
				self._name_to_top_map['bbox_inside_weights'] = idx
				idx += 1

#				top[idx].reshape(1, self._num_classes * 4)
				top[idx].set_shape((None, self._num_classes * 4))
				self._name_to_top_map['bbox_outside_weights'] = idx
				idx += 1

		print 'RoIData: name_to_top:', self._name_to_top_map
		assert len(top) == len(self._name_to_top_map)

	def forward(self, top):
		blobs = self._get_next_minibatch()
		
		feeder = []
		feeder_dict = {}
		i = 0
		for blob_name, blob in blobs.iteritems():
			top_ind = self._name_to_top_map[blob_name]
#			top[top_ind].reshape(*(blob.shape))
			top[top_ind].set_shape(blob.shape)
			feeder.append(blob.astype(np.float32, copy=False))
			feeder_dict[top_ind] = i
			i += 1
		
		return feeder, feeder_dict

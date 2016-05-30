import sys
sys.path.append("../tools")
import _init_paths
import caffe
import cPickle as pickle
import pdb

net = caffe.Net("/home/exon/py-faster-rcnn/models/ZF/fast_rcnn/train.prototxt", "/home/exon/py-faster-rcnn/output/default/voc_2007_trainval/zf_fast_rcnn_iter_40000.caffemodel", caffe.TEST)

weight_list = []
bias_list = []
name_dict = {}

i = 0
for param_name in net.params.keys():
	weight = net.params[param_name][0].data
	bias = net.params[param_name][1].data
	
	pdb.set_trace()
	
	weight_list.append(weight)
	bias_list.append(bias)
	
	name_dict[param_name] = i
	i += 1

pickle.dump(weight_list, open("weight_list.pkl", "wb"))
pickle.dump(bias_list, open("bias_list.pkl", "wb"))


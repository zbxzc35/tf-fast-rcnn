import cPickle as pickle
import pdb

def get_params():
	
	name_dict = {'bbox_pred': 8, 'fc6': 5, 'fc7': 6, 'cls_score': 7, 'conv3': 2, 'conv2': 1, 'conv1': 0, 'conv5': 4, 'conv4': 3}
	
	weight_list = pickle.load(open("weight_list.pkl", "rb"))
	bias_list = pickle.load(open("bias_list.pkl", "rb"))
	
	return weight_list, bias_list, name_dict

# python module, exposed to the outside callers
# here we accept only clean problem instances
# the caller should massage the data into what we expect.
# 
# In training, load a problem instance, call the learner, return the model.
# In prediction, load the data, load the model, output the prediction.
#
# exposed as functions/tools, not as objects/instances.
# 

# from theano_learner import TheanoLearner
# from theano_learner_v2 import TheanoLearner
from theano_learner_v3 import TheanoLearner
from ge_util import predict_instance
import json


# this include parameters, hyper-parameters, and learning knobs
class GE_data (object):
	def __init__(self):
		### raw data, in sparse format
		# dat['doc_id'] = {'fea_id': val}
		self.dat = {}
		self.labels = []
		
		### supervision signals
		# labeled_instances['doc_id'] = 'label'
		self.labeled_instances = {}
		# labeled_features['fea_id'] = {'label': prob}
		self.labeled_features = {}
		# label_prior['label'] = prob
		self.label_prior = {}

class GE_model (object):
	def __init__(self):
		### model parameters
		# weight['label'] = {fea_id: val}
		self.weight = {}
		# bias['label'] = val
		self.bias = {}

class GE_param (object):
	def __init__(self):
		# model hyper-parameters
		self.use_bias = False # add a constant feature
		self.l2_lambda = 0.01
		self.feature_lambda = 1.0
		self.use_initial_model = False

# data should be preloaded according to GE_data
# params should be set, according to GE_param
def GE_learn(data, model, param):
	l = TheanoLearner(data, model, param) # can be other learners
	train = l.get_train_function()
	l.learn(train)
	model = l.get_model()
	return model

def GE_get_train_function(data, model, param):
	l = TheanoLearner(data, model, param) # can be other learners
	train = l.get_train_function()
	return train

def GE_train(data, model, param, train):
	l = TheanoLearner(data, model, param) # can be other learners
	l.learn(train)
	model = l.get_model()
	return model

# dat['doc_id'] = {'fea_id': val}
def GE_predict(dat, model):
	pred = {}
	for doc_id, fea_dic in dat.items():
		prob = predict_instance(model.weight, model.bias, fea_dic)
		pred[doc_id] = prob
	return pred

# save and load model as a json format.
# (we can also do it in liblinear format)
def GE_save_model(model, model_path):
	f = open(model_path, 'w')
	m = {'weight': model.weight, 'bias': model.bias}
	json.dump(m, f, sort_keys=True, indent=2)
	f.close()

def GE_load_model(model_path):
	f = open(model_path)
	m = json.load(f)
	f.close()
	model = GE_model()
	model.weight = m['weight']
	model.bias = m['bias']
	return model




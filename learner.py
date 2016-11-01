# base class
# 
# given data and hyper parameter settings,
# learner will fit a logistic regression model,
# then return the parameters.
# 
# The core problem of learning is to define and compute
# the objective function, its gradients, etc.
# gradient can be computed by manual derivation, theano, tensorflow.
# 
# learner can call different optimization techniques,
# such as L-BFGS, gradient descent, SGD, or other routines.
# a learner should know what optimization method to use.
# 
# therefore, here the learner is an abstract base class.
# under the hood, we can have a theano_learn, a tensorflow learner
# and many other types of learners.

# import sys
import ge

def infer_labels(data):
	if len(data.labels) > 0:
		return data.labels
	if len(data.label_prior) > 0:
		return data.label_prior.keys()
	if len(data.labeled_features) > 0:
		return data.labeled_features.values()[0].keys()
	if len(data.labeled_instances) > 0:
		labels = {}
		for l in data.labeled_instances.values():
			if l not in labels:
				labels[l] = 1
		return labels.keys()
	exit('Please specify the set of labels through supervision!')

class Learner(object):
	def __init__(self, data, init_model, param):
		# record the pointer
		self.data = data
		self.init_model = init_model
		self.param = param
		self.labels = infer_labels(data)
		self.final_model = {}

	def get_model(self):
		m = ge.GE_model()
		m.weight = self.final_model
		m.bias = dict.fromkeys(self.labels, 0.0)
		return m

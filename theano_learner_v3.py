# one vs. rest logistic regression
# when #classes=2, only 1 set of parameters
# when #classes>2, we have '#classes' sets of parameters
# it implements the one vs. rest strategy in liblinear:
# http://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf

from learner import Learner
import scipy.sparse as sp
import theano.tensor as T
import numpy as np
from theano import sparse
import theano
import sys
from theano.ifelse import ifelse

def get_label_map(labels):
	m = {}
	for i in range(len(labels)):
		m[labels[i]] = i
	return m

def get_feature_map(dat):
	m = {}
	i = 0
	for doc_id, fea_dic in dat.items():
		for f, v in fea_dic.items():
			if f not in m:
				m[f] = i
				i += 1
	return m

def get_num_param_vecs(labels):
	if len(labels) == 2:
		return 1
	return len(labels)

# construct N x V matrix: N=number of data points, V=number of features
def construct_sparse_feature_matrix(dat, fea_map):
	num_row = 0
	row = []
	col = []
	val = []
	for doc_id, fea_dic in sorted(dat.iteritems(), key = lambda x: x[0]): # keep the same order in iterating through the data set: 0,...,N-1
		for f, v in fea_dic.items():
			row.append( num_row )
			col.append( fea_map[f] )
			val.append( v )
		num_row += 1
	spmat = sp.coo_matrix((val, (row, col)), shape=(num_row, len(fea_map))).tocsr()
	return spmat

# construct K x N mask matrix: K=number of labeled features, N=number of data points
# ** we can skip the labeled documents, if any. needs further experiments **
def construct_feature_document_indicators(dat, labeled_features):
	if len(labeled_features) == 0:
		return sp.coo_matrix(([1.], ([0], [0])), shape=(1, len(dat))).tocsr()

	num_col = 0
	row = []
	col = []
	val = []
	for doc_id, fea_dic in sorted(dat.iteritems(), key = lambda x: x[0]):
		num_row = 0
		for fea, label_dist in sorted(labeled_features.iteritems(), key = lambda x: x[0]):
			if fea in fea_dic:
				row.append( num_row )
				col.append( num_col )
				val.append( 1. )
			num_row += 1
		num_col += 1
	spmat = sp.coo_matrix((val, (row, col)), shape=(len(labeled_features), len(dat))).tocsr()
	return spmat

# construct feature expected distribution: K x C matrix: K=number of labeled features, C=number of classes
def construct_feature_expectation(labeled_features, label_map, labels):
	if len(labeled_features) == 0:
		return np.zeros((1, len(labels)))

	feature_expectation = []
	for fea, label_dist in sorted(labeled_features.iteritems(), key = lambda x: x[0]):
		dist = [1e-10]*len(labels)
		for l in labels:
			dist[ label_map[l] ] = label_dist[l]
		feature_expectation.append(dist)
	feature_expectation = np.asarray(feature_expectation)
	return feature_expectation

# construct L x N matrix: L=number of labeled documents, N=number of documents
def construct_label_document_indicators(dat, labeled_instances):
	if len(labeled_instances) == 0:
		return sp.coo_matrix(([1.], ([0], [0])), shape=(1, len(dat))).tocsr()

	num_row = 0
	num_col = 0
	row = []
	col = []
	val = []
	for doc_id, fea_dic in sorted(dat.iteritems(), key = lambda x: x[0]):
		if doc_id in labeled_instances:
			row.append( num_row )
			col.append( num_col )
			val.append( 1. )
			num_row += 1
		num_col += 1
	spmat = sp.coo_matrix((val, (row, col)), shape=(len(labeled_instances), len(dat))).tocsr()
	return spmat

# construct L x C matrix: L=number of labeled documents, C=number of classes
def construct_label_target(labeled_instances, label_map, labels):
	if len(labeled_instances) == 0:
		return np.zeros((1, len(labels)))

	target = []
	for doc_id, l in sorted(labeled_instances.iteritems(), key = lambda x: x[0]):
		dist = [0]*len(labels)
		dist[ label_map[l] ] = 1.
		target.append(dist)
	target = np.asarray(target)
	return target

class TheanoLearner(Learner):
	def __init__(self, data, init_model, param):
		Learner.__init__(self, data, init_model, param)

		# print 'Hey, TheanoLearner is initializing!'
		### mapping from input data to continuous integer indices 
		# label_set['label'] = j
		self.label_map = get_label_map(self.labels)
		# feature_set['fea_id'] = k
		self.feature_map = get_feature_map(data.dat)
		self.num_param_vecs = get_num_param_vecs(self.labels)

	# we separate the part building a training function from the actual learning.
	# before learning happens, we have to have the training function.
	# that's the trick to learn from different data sets. 
	def get_train_function(self):
		# specify the computational graph
		num_param_vecs = T.scalar('num_param_vecs')
		# weight = theano.shared(np.random.randn(len(self.feature_map), self.num_param_vecs), name='weight')
		weight = theano.shared(np.zeros((len(self.feature_map), self.num_param_vecs)), name='weight')
		feat_mat = sparse.csr_matrix(name='feat_mat')
		pred = T.nnet.sigmoid( sparse.dot(feat_mat, weight) ) # one-vs-rest

		o_pred = ifelse(T.gt(self.num_param_vecs, 1), pred / pred.sum(axis=1).reshape((pred.shape[0], 1)), T.concatenate( [pred, 1-pred], axis=1 ) )  

		f_target = T.matrix('f_target')
		f_mask_mat = sparse.csr_matrix(name='f_mask_mat')
		f_sum_pred = sparse.dot( f_mask_mat, o_pred )
		f_pred = f_sum_pred / f_sum_pred.sum(axis=1).reshape((f_sum_pred.shape[0], 1))

		i_target = T.matrix('i_target')
		i_mask_mat = sparse.csr_matrix(name='l_mask_mat')
		i_pred = sparse.dot( i_mask_mat, pred )

		# objective = self.param.feature_lambda * T.nnet.categorical_crossentropy(f_pred, f_target).sum() + T.nnet.binary_crossentropy(i_pred, i_target).sum() + self.param.l2_lambda * (weight ** 2).sum() / 2
		objective = 0.0 * T.nnet.categorical_crossentropy(f_pred, f_target).sum() + T.nnet.binary_crossentropy(i_pred, i_target).sum() + self.param.l2_lambda * (weight ** 2).sum() / 2

		grad_weight = T.grad(objective, weight)

		# print 'Compiling function ...'
		# compile the function
		train = theano.function(inputs = [feat_mat, f_mask_mat, f_target, i_mask_mat, i_target], outputs = [objective, weight], updates=[(weight, weight - 0.1*grad_weight)] )

		return train

	def learn(self, train):
		# print 'hey I am learning!'
		sp_feature = construct_sparse_feature_matrix(self.data.dat, self.feature_map)
		
		fea_doc_ind = construct_feature_document_indicators(self.data.dat, self.data.labeled_features)
		feature_expectation = construct_feature_expectation(self.data.labeled_features, self.label_map, self.labels)

		lbl_doc_ind = construct_label_document_indicators(self.data.dat, self.data.labeled_instances)
		lbl_target = construct_label_target(self.data.labeled_instances, self.label_map, self.labels)
		
		w = None
		old_obj = -1
		for i in range(20000):
			obj, w = train(sp_feature, fea_doc_ind, feature_expectation, lbl_doc_ind, lbl_target)
			if abs(obj - old_obj) < 5e-4: # stop training when the objective does not change much
				break
			old_obj = obj
			# if i % 10 == 0:
			# 	sys.stderr.write('{}\t{}\n'.format(i, obj))

		# print 'obj', obj
		# print 'prd', prd

		raw_model = w
		# translate raw model
		model = {}
		for lbl in self.labels:
			model[lbl] = {}
		for fea, row_idx in self.feature_map.items():
			for lbl, col_idx in self.label_map.items():
				model[lbl][fea] = raw_model[row_idx, col_idx]

		self.final_model = model

		
		

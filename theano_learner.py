from learner import Learner
import scipy.sparse as sp
import theano.tensor as T
import numpy as np
from theano import sparse
import theano
import sys

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

def extract_rows_containing_feature(dat, fea, fea_map):
	num_row = 0
	row = []
	col = []
	val = []
	for doc_id, fea_dic in dat.items():
		if fea in fea_dic: # add the entire row
			for f, v in fea_dic.items():
				row.append( num_row )
				col.append( fea_map[f] )
				val.append( v )
			num_row += 1
	spmat = sp.coo_matrix((val, (row, col)), shape=(num_row, len(fea_map))).tocsr()
	return spmat, num_row

class TheanoLearner(Learner):
	def __init__(self, data, init_model, param):
		Learner.__init__(self, data, init_model, param)

		# print 'Hey, TheanoLearner is initializing!'
		### mapping from input data to continuous integer indices 
		# label_set['label'] = j
		self.label_map = get_label_map(self.labels)
		# feature_set['fea_id'] = k
		self.feature_map = get_feature_map(data.dat)

	# we separate the part building a training function from the actual learning.
	# before learning happens, we have to have the training function.
	# that's the trick to learn from different data sets. 
	def get_train_function(self):
		# specify the computational graph
		target = T.matrix('target')
		weight = theano.shared(np.random.randn(len(self.feature_map), len(self.label_map)), name='weight')
		feat_mat = sparse.csr_matrix(name='feat_mat')
		mask_mat = sparse.csr_matrix(name='mask_mat')
		sum_pred = sparse.dot( mask_mat, T.nnet.softmax( sparse.dot(feat_mat, weight) ) )
		pred = sum_pred / sum_pred.sum(axis=1).reshape((sum_pred.shape[0], 1))
		objective = T.nnet.categorical_crossentropy(pred, target).sum() + self.param.l2_regularization * (weight ** 2).sum()
		grad_weight = T.grad(objective, weight)

		# print 'Compiling function ...'
		# compile the function
		train = theano.function(inputs = [feat_mat, mask_mat, target], outputs = [objective, weight], updates=[(weight, weight - 0.1*grad_weight)] )

		return train

	def learn(self, train):
		# print 'hey I am learning!'
		# construct K matrices for K labeled features
		mats = []
		block_size = []
		for fea in self.data.labeled_features:
			mat, num_row = extract_rows_containing_feature(self.data.dat, fea, self.feature_map)
			mats.append(mat)
			block_size.append(num_row)
		stack_mat = sp.vstack(mats, format='csr')

		# construct the mask matrix
		row = []
		col = []
		val = []
		accu_c = 0
		for r in range(len(block_size)):
			for c in range(block_size[r]):
				row.append(r)
				col.append(c + accu_c)
				val.append(1.)
			accu_c += block_size[r]
		mask_mat = sp.coo_matrix((val, (row, col)), shape=(len(block_size), accu_c)).tocsr()

		# construct target: labeled features
		target = []
		for fea, label_dist in self.data.labeled_features.items():
			dist = [1e-10]*len(self.labels)
			for l in self.labels:
				dist[ self.label_map[l] ] = label_dist[l]
			target.append(dist)

		target = np.asarray(target)
		# print target

		w = None
		old_obj = -1
		for i in range(20000):
			obj, w = train(stack_mat, mask_mat, target)
			if abs(obj - old_obj) < 5e-4: # stop training when the objective does not change much
				break
			old_obj = obj
			# if i % 1000 == 0:
			# 	sys.stderr.write('{}\t{}\r'.format(i, obj))

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

		
		

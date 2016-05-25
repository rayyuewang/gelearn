# command line wrappers, assuming the tool is used from the command line
# here one should use the argparse module, to allow flexible inputs.
# 
# In training, we will create a clean problem instance for the learner.
# In prediction, we will pass the data to the learned model.

import sys, os
import argparse
from ge import *
import util

def parse_arg_learn():
	parser = argparse.ArgumentParser(prog='GE learn')
	# positional arguments
	parser.add_argument('learn', help = '"learn": learn or update an exponential model')
	parser.add_argument('data', help = 'data examples. line format: task_id data_id [feature_id:value]+')
	parser.add_argument('model', help = 'file path for the final model')
	# optional arguments
	parser.add_argument('-f', '--labeled_features', help = 'file path for the labeled features')
	parser.add_argument('-l', '--labeled_instances', help = 'file path for the labeled instances')
	parser.add_argument('-p', '--label_prior', help = 'prior distribution of labels')
	parser.add_argument('-i', '--initial_model', help = 'an existing model for initialization')
	parser.add_argument('--l2', type = float, help = 'L-2 regularization weight')
	
	args = parser.parse_args()

	# validate args
	if (not args.labeled_features) and (not args.labeled_instances):
		exit ('Please specify either labeled feature or labeled instance. label prior can be added as additional supervision.')
	if not os.path.isfile(args.data):
		exit ('Data file not found: {}. Please specify a valid data example path.'.format(args.data))
	if args.labeled_features and not os.path.isfile(args.labeled_features):
		exit ('Labeled feature file not found: {}. Please specify a valid labeled feature path.'.format(args.labeled_feature))

	return args

def parse_arg_predict():
	parser = argparse.ArgumentParser(prog='GE predict')
	parser.add_argument('predict', help = '"predict": predict data using a learned model')
	parser.add_argument('data', help = 'data examples. line format: task_id data_id [feature_id:value]+')
	parser.add_argument('model', help = 'file path for the input model')
	parser.add_argument('output', help = 'file path for the output prediction')

	args = parser.parse_args()

	# validate args
	if not os.path.isfile(args.data):
		exit ('Data file not found: {}. Please specify a valid data example path.'.format(args.data))
	if not os.path.isfile(args.model):
		exit ('Model file not found: {}. Please specify a valid model path.'.format(args.model))

	return args

# parse arguments, get data, (initial) model, and parameters
def ge_cmd_learn():
	args = parse_arg_learn()
	
	# prepare input to GE_learn
	data = GE_data()
	data.dat = util.load_data(args.data)
	data.labeled_features = util.load_labeled_features(args.labeled_features)
	init_model = GE_model()
	param = GE_param()
	if args.l2:
		param.l2_regularization = args.l2
	final_model_path = args.model

	# print data

	final_model = GE_learn(data, init_model, param)
	util.save_model(final_model, final_model_path)
	return

# parse arguments, get data and model, output prediction
def ge_cmd_predict():
	args = parse_arg_predict()

	# prepare input to GE_learn
	data = util.load_data(args.data)
	model = util.load_model(args.model)
	pred_path = args.output

	pred = GE_predict(data, model)
	util.write_prediction(pred, pred_path)
	return

if __name__ == '__main__':
	if len(sys.argv) < 2:
		exit ('First argument should be "learn" or "predict".\nTo see options, use "learn -h" or "predict -h".')
	if sys.argv[1] == 'learn':
		ge_cmd_learn()
	elif sys.argv[1] == 'predict':
		ge_cmd_predict()
	else:
		exit ('First argument should be "learn" or "predict".\nTo see options, use "learn -h" or "predict -h".')


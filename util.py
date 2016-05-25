# Here we can create some useful tools: data manipulation, write to file, etc.
# Prediction is here. the function might be used in many places.

from math import exp
import json
import ge

def predict_instance(weight, bias, fea_dic):
	s = {}
	for l in bias:
		s[l] = bias[l] + 1e-10
		for f, v in fea_dic.items():
			if f in weight[l]:
				s[l] += weight[l][f] * v
	mean = sum(s.values()) / len(s)
	for l in s:
		s[l] = exp(s[l] - mean)
	normalizer = sum(s.values())
	for l in s:
		s[l] /= normalizer
	return s

def load_data(data_path):
	dat = {}
	in_f = open(data_path)
	for line in in_f:
		task_id, pmid, feature_line = line.split('\t')
		d = {}
		for fv in feature_line.strip().split(' '):
			f, v = fv.split(':')
			d[f] = float(v)
		dat[pmid] = d
	in_f.close()
	return dat

# each line is in the same surface format as data
# task_id feature_id [label:value]+
# The BIG CAVEAT is: label:value shall be a probability distribution!
def load_labeled_features(labeled_feature_path):
	return load_data(labeled_feature_path)

def write_prediction(pred, pred_path):
	if len(pred) == 0:
		return
	label_list = pred.values()[0].keys()
	ou_f = open(pred_path, 'w')
	
	# write predictions
	for doc_id, label_prob in pred.items():
		predicted_label = sorted(label_prob.items(), key = lambda x: x[1], reverse = True)[0][0]
		ou_f.write('{}\t{}\t'.format(doc_id, predicted_label))
		for label in label_list:
			ou_f.write('{}:{:.6f} '.format(label, label_prob[label]))
		ou_f.write('\n')
	ou_f.close()


# save and load model as a json format.
# (we can also do it in liblinear format)
def save_model(model, model_path):
	f = open(model_path, 'w')
	m = {'weight': model.weight, 'bias': model.bias}
	json.dump(m, f, sort_keys=True, indent=2)
	f.close()

def load_model(model_path):
	f = open(model_path)
	m = json.load(f)
	f.close()
	model = ge.GE_model()
	model.weight = m['weight']
	model.bias = m['bias']
	return model

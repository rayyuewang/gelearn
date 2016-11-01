from ge import *

dat = {'doc_1': {'1':1., '2':3.}, 'doc_2': {'1':2., '3':4.}, 'doc_3': {'2':1., '3':0.5}, 'doc_4': {'3':2., '4':0.8} }
# labeled_features = {'1': {'c1':0.9, 'c2':0.1}, '2': {'c1':0.8, 'c2':0.2}, }
labeled_instances = {'doc_2': 'c2', 'doc_1': 'c1'}

# prepare input for GE_learn
data = GE_data()
data.dat = dat
data.labels = ['c1', 'c2', 'c3']
# data.labeled_features = labeled_features
# data.labeled_instances = labeled_instances

old_model = GE_model()
param = GE_param()

# use GE_learn to train a model
new_model = GE_learn(data, old_model, param)

# use GE_predict to predict labels
pred = GE_predict(dat, new_model)

print pred

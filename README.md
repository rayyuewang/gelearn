# gelearn: Generalized Expectation Learning

This tool learns a logistic regression model using the [generalized expectation objective] (https://people.cs.umass.edu/~mccallum/papers/druck08sigir.pdf) by Druck, Mann, and McCallum. Usually, logistic regression models are trained by minimizing the cross-entropy objective, which requires labeled instances. Using generalized expectation (GE) objective, we can train logistic regression models by just labeling features. This allows the user to quickly transfer domain knowledge in rapid classifier building, saving labeling effort and alleviating the _cold start_ problem.


Dependency
----------
* Python (>= 2.7.3)
* Theano (>= 0.8.2)

Usage
-----
This tool comes with both a command-line interface and a Python module interface. Its usage pattern is similar to that of LIBLINEAR.

### Command-line interface

__Learn logistic regression model__

`python /path/to/ge_cmd.py learn [data] [model] -f [labeled_features]`

Each line of the `data` file is an unlabeled feature vector in sparse format:

`[data_id] TAB ([feature_id]:value )+`

* `data_id`: string identifier for the data point.
* `feature_id`: string identifier for the feature dimension, need not be an integer

Each line of the `labeled_features` file is a probability distribution of labels given a feature:

`[feature_id] TAB ([label_id]:Pr(label_id|feature_id) )+`

* `label_id`: string identifier for a class label
* Pr(labe)

Predict instances using learned model
`python /path/to/ge_cmd.py predict [data] [model] [output]`

For more information, type
`python /path/to/ge_cmd.py learn -h`
`python /path/to/ge_cmd.py predict -h`



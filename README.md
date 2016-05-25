# gelearn

GE (generalized expectation) Learn
==================================

This tool learns a logistic regression model using the [generalized expectation objective] (https://people.cs.umass.edu/~mccallum/papers/druck08sigir.pdf) by Druck, Mann, and McCallum. Usually, logistic regression models are trained by minimizing the cross-entropy objective, which requires labeled instances. Using generalized expectation (GE) objective, we can train logistic regression models by just labeling features. This allows the user to quickly transfer domain knowledge in rapid classifier building, saving labeling effort and alleviating the _cold start_ problem.





Dependency
----------
* Python (>= 2.7.3)
* Theano (>= 0.8.2)

Usage
=====
This tool comes with both a command-line interface and a Python module interface. Its usage pattern is similar to that of LIBLINEAR. 

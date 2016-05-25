#!/bin/bash

echo 'Training ...'
python ../ge_cmd.py learn test.dat test.model -f test.labeled_feature

echo 'Predicting ...'
python ../ge_cmd.py predict test.dat test.model test.pred


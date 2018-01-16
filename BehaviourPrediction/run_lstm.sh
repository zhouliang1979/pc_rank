#!/bin/sh


dim_hiden=10
all_dataset=train_1500w

python balance_data.py $all_dataset train_set 20000
python balance_data.py $all_dataset test_set 300

python -u lstm.py train_set test_set lstm_model lstm_model_npz $dim_hiden

#python -u check_ProbDistr.py test_set lstm_model_npz.npz prob_test_set $dim_hiden



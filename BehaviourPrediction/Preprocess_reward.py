__author__ = 'cairui'
import cPickle
import gzip
import os

import numpy
import theano
import numpy as np

def get_dataset(data_path):
    
    print("load data :", data_path)
    All_data = open(data_path).read().splitlines()
    train_samples = []
    train_samples_x = []
    train_samples_y = []
    reward_number = 0
    noreward_number = 0

    level_1 = 0
    total_sum = 0

    for line in All_data:
        total_sum += 1
        train_sample_x = []
        train_sample_y = []

        user_id, power_tags_str, brand_tags_str, rewards_tags_str = line.split('\t')

        power_tags = power_tags_str.split(',')
        brand_tags = brand_tags_str.split(',')
        rewards_tags = rewards_tags_str.split(',')

        if rewards_tags[-1] == '0_0_0':
            noreward_number += 1
        else:
            reward_number += 1

        if power_tags[-1] == '101':
            level_1 += 1

        for i in range(0, len(power_tags)):
            single_behavior = []
            single_behavior.append(power_tags[i])
            single_behavior.append(brand_tags[i])
            single_behavior.append(rewards_tags[i])

            if i == len(power_tags)-1:
                train_sample_y = single_behavior
            else:
                train_sample_x.append(single_behavior)
        train_samples_x.append(train_sample_x)
        train_samples_y.append(train_sample_y)

    train_samples.append(train_samples_x)
    train_samples.append(train_samples_y)
    print("the reward number is:", reward_number)
    print("the no reward number is", noreward_number)
    print("total power_tags is:", total_sum)
    print("level 1 power_tags is:", level_1)
    print(float(level_1)/total_sum)
    return train_samples


#train = get_dataset()

#print(len(train))
#print(train[0][1])
#print(train[1][1])

#print(train[0][2])
#print(train[1][2])

#print(train[0][3])
#print(train[1][3])


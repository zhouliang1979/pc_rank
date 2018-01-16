__author__ = 'cairui'
import cPickle
import gzip
import os
import sys

import numpy
import theano
import numpy as np

def balance_dataset(all_dataset, train_set, base_num=20000):

    print("all_dataset:", all_dataset)
    print("train_set:", train_set)

    balance_file = open(train_set, 'w')
    All_data = open(all_dataset).read().splitlines()
    
    reward_number = 0
    noreward_number = 0

    level_1 = 0
    level_num = [0,0,0,0,0,0,0]
    level_limit = [1*base_num,1*base_num,2*base_num,3*base_num,2*base_num,1*base_num,1*base_num]
    level_noReward_num = [0,0,0,0,0,0,0]
    total_sum = 0

    for line in All_data:
        if np.random.rand() >0.5:
            continue
        train_sample_x = []
        train_sample_y = []
        user_id,  power_tags_str, brand_tags_str, rewards_tags_str = line.split('\t')
        power_tags = power_tags_str.split(',')
        brand_tags = brand_tags_str.split(',')
        rewards_tags = rewards_tags_str.split(',')

        if len(power_tags)<4:
            continue

        ## remove the data point with 2 '0' tags
        if power_tags[-1]=='100' or brand_tags[-1]=='310':
            continue

        if level_num[int(power_tags[-1]) - 101] >= level_limit[int(power_tags[-1]) - 101]:
            continue
        
        if rewards_tags[-1] == '0_0_0':
            if level_noReward_num[int(power_tags[-1]) - 101] >= level_limit[int(power_tags[-1]) - 101]/2.0 or level_num[int(power_tags[-1]) - 101] >= level_limit[int(power_tags[-1]) - 101] :
                continue
            level_noReward_num[int(power_tags[-1]) - 101] += 1
            noreward_number += 1
        else:
            reward_number += 1

        if power_tags[-1] == '101':
            level_1 += 1
        total_sum += 1
        level_num[int(power_tags[-1]) - 101] += 1
      
        balance_file.write(line)
        balance_file.write('\n')
   
    print("the reward number is:", reward_number)
    print("the no reward number is", noreward_number)
    print("total power_tags is:", total_sum)
    print("level 1 power_tags is:", level_1)
    print(level_num)
    print(float(level_1)/total_sum)



if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("balance_data.py all_dataset train_set base_num")
    else:
        print("start balance_data:")
        all_dataset=sys.argv[1]
        train_set=sys.argv[2]
        base_num=int(sys.argv[3])
        balance_dataset(all_dataset, train_set, base_num)


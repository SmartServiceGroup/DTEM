import os
import sys
import json

import torch
from torch.utils.data import random_split

if __name__ == "__main__":
    data_path = "./data/user_watch_repos.json"
    K = 10
    
    with open(data_path, "r", encoding="utf-8") as inf:
        samples = json.load(inf)
    
    single_fold_length = int(len(samples) * 0.1)
    fold_lengths = [single_fold_length] * 9
    fold_lengths.append(len(samples) - 9 * single_fold_length)
    
    split_samples = random_split(list(samples), fold_lengths, generator=torch.Generator().manual_seed(42))
    
    for i in range(K):
        test_samples = list(split_samples[i])
        train_samples = []
        test_path = "./data/10fold/test{}.json".format(i)
        train_path = "./data/10fold/train{}.json".format(i)

        # region 
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        # endregion
        
        for j in range(K):
            if i == j:
                continue
            else:
                train_samples.extend(list(split_samples[j]))
                    
        with open(train_path, "w", encoding="utf-8") as ouf:
            json.dump(train_samples, ouf)
        with open(test_path, "w", encoding="utf-8") as ouf:
            json.dump(test_samples, ouf)
        
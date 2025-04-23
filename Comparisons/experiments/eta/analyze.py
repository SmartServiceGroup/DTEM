#!/usr/bin/env python3

# %% 

import pickle 

with open("./result.pkl", "rb") as inf:
    data = pickle.load(inf)


# %% 

def rate(data, thres=10): 
    size = len(data)  # 176844 
    data.sort(key=lambda x: x[1], reverse=True)
    if size == 0: 
        return 1
    for i, (_, v) in enumerate(data):
        if v <= thres:
            break
    return i / size

def flattern_rate(data, thres=10): 
    values = [it for repo in data.values() for it in repo]
    return rate(values, thres)


# %% 

print(len(data))  # 5000
print(flattern_rate(data))  # 11.74%, Yeah! 
print(flattern_rate(data, thres=9))  # 11.74%, Yeah! 


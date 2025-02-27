import os
import json
import numpy as np

def analysis(topks):
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    mrr = 0

    top1 += (sum(topks[:2]) != 0)
    top3 += (sum(topks[:4]) != 0)
    top5 += (sum(topks[:6]) != 0)
    top10 += (sum(topks[:11]) !=0)

    for i in range(1, len(topks)):
        if topks[i] > topks[i - 1]:
            mrr += 1 / i
            break
    return top1, top3, top5, top10, mrr

"""
## Model RESULT
0.121 0.298 0.414 0.618 0.260

## on modified dataset
0.12998712998713 0.305019305019305 0.4491634491634492 0.703989703989704 0.2850200997425676
27.931788931788933
"""

"""
# model_2entities on modified dataset
0.11454311454311454 0.2972972972972973 0.43114543114543114 0.7052767052767053 0.27353745157261283
27.931788931788933
"""

if __name__ == "__main__":
    src_file = "./result/model_2entities_result_valid_test_modified.json"
    with open(src_file, "r", encoding="utf-8") as inf:
        src = json.load(inf)
    
    total = len(src)
    scope_length = 0

    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0
    MRR = 0

    for item in src:
        topks = src[item][0]
        tmp1, tmp3, tmp5, tmp10, mrr = analysis(topks)
        top1 += tmp1
        top3 += tmp3
        top5 += tmp5
        top10 += tmp10
        MRR += mrr
        scope_length += len(src[item][1])

    print(top1 / total, top3 / total, top5 / total, top10 / total, MRR / total)
    print(scope_length / total)
import os
import json
import numpy as np
import sys

def analysis(topks):
    top1  = 0
    top3  = 0
    top5  = 0
    top10 = 0
    mrr   = 0  # MRR: https://blog.csdn.net/jiangjiang_jian/article/details/108246103

    top1  += (sum(topks[:2])  != 0)
    top3  += (sum(topks[:4])  != 0)
    top5  += (sum(topks[:6])  != 0)
    top10 += (sum(topks[:11]) != 0)

    for i in range(1, len(topks)):
        if topks[i] > topks[i - 1]:
            mrr += 1 / i
            break
    return top1, top3, top5, top10, mrr

def main(src_file):

    print(src_file)

    # src_file = "./result/model_result_valid_test.json"
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
        top1    += tmp1
        top3    += tmp3
        top5    += tmp5
        top10   += tmp10
        MRR     += mrr
        scope_length += len(src[item][1])
        

    print(top1 / total, top3 / total, top5 / total, top10 / total, MRR / total)
    print(scope_length / total)

    print(
        f'{top1/total:.3f} | {top3/total:.3f} | {top10/total:.3f} | {MRR/total:.3f}'
    )
    print()


if __name__ == "__main__":
    files = [
        # 'result/noise/HR/0.7/ContributionRepo.txt',
        'result/noise/HR/0.7/PRReviewer.txt',
        # 'result/noise/HR/0.7/RepoMaintainer.txt',

        # 'result/noise/HR/1.0/ContributionRepo.txt',
        'result/noise/HR/1.0/PRReviewer.txt',
        # 'result/noise/HR/1.0/RepoMaintainer.txt',
    ]
    for it in files: 
        main(it)
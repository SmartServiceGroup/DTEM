#!/usr/bin/env python3

from RecommendationTasks.ContributionRepo_CF.config import \
        CF_DICT_FILE_PATH, VALID_TEST_DATA_FILE_PATH, VALID_RESULT_PATH, \
        load_data

from RecommendationTasks.ContributionRepo_CF.model  \
    import CollaborativeFiltering

import json
from typing import List, Dict, Tuple


def evaluate(model_postfix='full'): 
    '''
        这个函数用于评估协同过滤(CF)模型的效果. 模型的选择由 model_postfix 指定.
        会生成一个中间文件. 该文件放在 ./result 中. 

        model_postfix 的可选项可以在 
        RecommendationTasks/ContributionRepo_CF/bin/
        下找到. 
        比如如果存在文件 cf_dict.pkl.top40, 则可用的 postfix 为 top40 

        see also: 
            ../model.py -> train_model()
            ./metric/metric.py -> metric()
    '''

    klee = CollaborativeFiltering()
    klee.load_pickle(CF_DICT_FILE_PATH + '.' + model_postfix)

    # @see: RecommendationTasks/ContributionRepo/README.md
    dataset = load_data(VALID_TEST_DATA_FILE_PATH)

    result: Dict[int, Tuple[List[int], List[int]]] = {}

    # dev_id: int
    # search_scope: List[int], 此开发者开发过的所有仓库(从图构建的). 
    # gt: List[int], ground truth. 在训练的过程中没有看到的仓库列表. 
    # 这个验证是希望说明, 在协同过滤(CF)中没有看到的仓库, 在此阶段也可以通过它在search_scope中搜索得到. 
    for dev_id, search_scope, gt in dataset: 
        if len(gt) < 5: continue

        # Exists dev_id, s.t. len(recs) < 20. Be careful. 
        # (sometimes happens when len(search_scope) < 20)
        # recs: recommendations
        recs: List[int] = klee.recommend(dev_id, search_scope)[:20]

        acc: List[int] = [0] * 21
        for i in range(20):
            acc[i + 1] = acc[i] + (i < len(recs) and recs[i] in gt)

        # print(recs)
        # print('\t' + str(acc))
        # print(len(recs))

        result[dev_id] = (acc, search_scope)

    with open(VALID_RESULT_PATH + '.' + model_postfix, 'w') as fp:
        json.dump(result, fp)
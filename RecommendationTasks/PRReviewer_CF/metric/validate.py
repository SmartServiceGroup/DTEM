#!/usr/bin/env python3

from re import search
from RecommendationTasks.PRReviewer_CF.config import \
        CF_DICT_FILE_PATH, VALID_TEST_DATA_FILE_PATH, VALID_RESULT_PATH, \
        load_data

from RecommendationTasks.PRReviewer_CF.model  \
    import CollaborativeFiltering

import json
from typing import List, Dict, Tuple


def evaluate(model_postfix='full', partial=False): 

    if partial: model_postfix += '.partial'
    klee = CollaborativeFiltering()
    klee.load_pickle(CF_DICT_FILE_PATH + '.' + model_postfix)

    dataset = load_data(VALID_TEST_DATA_FILE_PATH)

    result: Dict[int, Tuple[List[int], List[int]]] = {}
    
    cnt = 0
    for repo_id, pr_id, search_scope, gt in dataset:
        if len(search_scope) < 10: continue        
        cnt += 1
        
        recs: List[int] = klee.recommend(repo_id, search_scope)[:20]
        acc: List[int] = [0] * 21
        
        for i in range(20):
            acc[i + 1] = acc[i] + (i < len(recs) and recs[i] in gt)
        result[repo_id] = (acc, search_scope)
    
    with open(VALID_RESULT_PATH + '.' + model_postfix, 'w') as fp:
        json.dump(result, fp)
    print(str(cnt) + "recommendations finished")
path = "/media/dell/disk/vkx/DTEM"
import sys
sys.path.append(path)

from RecommendationTasks.SimDeveloper_CF.model import train_model
from RecommendationTasks.SimDeveloper_CF.metric.validate import evaluate
from RecommendationTasks.SimDeveloper_CF.metric.metric import metric

if __name__ == "__main__":
    topk = 0
    partial = True
    model_postfix = 'full'
    if topk > 0:
        model_postfix = 'top' + str(topk)
    
    train_model(top_count=topk, partial=partial)
    evaluate(model_postfix=model_postfix, partial=partial)
    metric(model_prefix=model_postfix, partial=partial)
    
'''
3494recommendations finished
Metrics of RecommendationTasks/SimDeveloper_CF/metric/result/result_valid_test.json.full.partial: 
0.6649 | 0.6972 | 0.7092 | 0.7138 | 0.6826

1303recommendations finished
Metrics of RecommendationTasks/SimDeveloper_CF/metric/result/result_valid_test_modified.json.full.partial: 
0.3400 | 0.4728 | 0.5081 | 0.5411 | 0.4137

'''
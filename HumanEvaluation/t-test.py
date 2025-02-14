import scipy.stats
import config
import numpy as np

tasks = config.RecommendationTasks
task2model = config.SUFFIX

model2task = {}
for task in task2model:
    for model in task2model[task]:
        model2task[model] = []
for task in task2model:
    for model in task2model[task]:
        model2task[model].append(task)

result = {}
for task in tasks:
    result[task] = {}
    for model in task2model[task]:
        result[task][model] = []

num_result = 10
num_data = 3        
for result_index in range(num_result):
    result_file = config.RESULT_PREFIX + "{}.txt".format(result_index)
    with open(result_file, 'r') as f:
        for task in tasks:
            for data_index in range(num_data):
                data = f.readline()
                data = [int(x) for x in data.split()]
                i = 0
                for model in task2model[task]:
                    result[task][model].extend([data[i], data[i + 1]])
                    i += 2
            f.readline()
            
# model_score_distribution = {}
# for task in result:
#     model_score_distribution[task] = {}
#     for model in result[task]:
#         model_score_distribution[task][model] = [0, 0, 0, 0, 0, 0]
    
# for task in result:
#     for model in result[task]:
#         for score in result[task][model]:
#             model_score_distribution[task][model][score] += 1
# print(model_score_distribution)

modelAvg = {}
for model in model2task:
    scores = []
    for task in result:
        if model in result[task]:
            scores.extend(result[task][model])
    modelAvg[model] = np.average(scores)
print(modelAvg)
    
for model in model2task:
    if model == "DTEM":
        continue
    print(model)
    modelScores = []
    DTEMScores = []
    for task in model2task[model]:
        if task == "PRReviewer":
            continue
        modelScores.extend(result[task][model])
        DTEMScores.extend(result[task]["DTEM"])
    ttest_result = scipy.stats.ttest_ind(DTEMScores, modelScores)
    print(ttest_result)

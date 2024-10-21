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
    modelScores = []
    DTEMScore = []
    for task in model2task[model]:
        modelScores.extend(result[task][model])
        DTEMScore.extend(result[task]["DTEM"])
        result = scipy.stats.ttest_ind(modelScores, DTEMScore)
        print(result)
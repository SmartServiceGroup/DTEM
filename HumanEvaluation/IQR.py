

import config, json
import numpy as np

tasks = config.RecommendationTasks
task2model = config.SUFFIX

num_results = 10
num_data = 3

result = {}
for task in tasks:
    result[task] = {}
    for model in task2model[task]:
        result[task][model] = {'data':[], 'iqr':None}

for result_index in range(num_results):
    result_file = config.RESULT_PREFIX + "{}.json".format(result_index)
    with open(result_file, 'r') as f:
        result_data = json.load(f)
    for task in tasks:
        for model in task2model[task]:
            task_model = task + "-" + model
            result[task][model]['data'].extend(result_data[task][model])
            
            
for task in tasks:
    for model in task2model[task]:
        print("%s, %s, %d" % (task, model, len(result[task][model])))
        data = result[task][model]['data']
        data = np.array(data)
        q1 = np.percentile(data, 25, interpolation='midpoint')
        q3 = np.percentile(data, 75, interpolation='midpoint')
        iqr = q3 - q1
        result[task][model]['iqr'] = [q1, q3, iqr]
        print(task_model, iqr)
        del result[task][model]['data']
with open(config.RESULT_PREFIX + "iqr.json", 'w') as f:
    json.dump(result, f, indent=2, separators=(',', ': '))
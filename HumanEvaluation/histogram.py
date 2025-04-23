

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
        result[task][model] = []

for result_index in range(num_results):
    result_file = config.RESULT_PREFIX + "{}.json".format(result_index)
    with open(result_file, 'r') as f:
        result_data = json.load(f)
    for task in tasks:
        for model in task2model[task]:
            task_model = task + "-" + model
            result[task][model].extend(result_data[task][model])
            
with open(config.RESULT_PREFIX + "histogram.json", 'w') as f:
    json.dump(result, f, indent=2, separators=(',', ': '))
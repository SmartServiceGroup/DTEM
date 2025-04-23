

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
        result[task][model] = {'data':[], 'mode':None}

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
        print("%s, %s, %d" % (task, model, len(result[task][model]['data'])), flush=True)
        data = result[task][model]['data']
        data = np.array(data)
        mode = int(np.argmax(np.bincount(data)))
        result[task][model]['mode'] = mode
        print(mode, flush=True)
        del result[task][model]['data']
with open(config.RESULT_PREFIX + "mode.json", 'w') as f:
    json.dump(result, f, indent=2, separators=(',', ': '))
import pandas as pd
import config
import pingouin as pg

tasks = config.RecommendationTasks
task2model = config.SUFFIX

scores = [[] for x in range(10)]

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
                    if task == "PRReviewer":
                        continue
                    else:
                        scores[result_index].extend([data[i], data[i + 1]])
                    i += 2
            f.readline()

num_of_subjects = len(scores[0])
score = [item for sublist in scores for item in sublist]

judge = []
for i in range(num_result):
    judge.extend([str(i)] * num_of_subjects)
    
subject = []
for i in range(num_result):
    subject.extend([str(j) for j in range(num_of_subjects)])
        

#create DataFrame
df = pd.DataFrame({'subject':subject,
                   'judge': judge,
                   'rating': score})


icc = pg.intraclass_corr(data=df, targets='subject', raters='judge', ratings='rating')
print(icc)
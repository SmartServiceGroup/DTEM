import json
random_users = []
with open('../data/users10000_random.json','r') as f:
    random_users = json.load(f)
count = 0
with open('../data/repo_contributions.txt','r') as repo_contributions:
    for line in repo_contributions:
        ru_list = line.strip().split('\t')
        repo = ru_list[0]
        users = json.loads(ru_list[1])
        flitered_users = []
        for user in users:
            if user[0] in random_users:
                count += user[1]
print(count)

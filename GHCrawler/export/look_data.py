import json


issue_path = '../cleaned/repo_issues.txt'
pr_path = '../cleaned/repo_prs.txt'
discussion_path = '../rawdata/repo_discussion.txt'
contributor_path = '../../GNN/DataPreprocess/full_graph/content/contributors.json'

issue_cnt = 0
pr_cnt = 0
repo_list = []
contributor_list = []
contributor_pr = {}

with open(contributor_path, 'r') as f:
    contributor_list = json.load(f)
contributor_list = contributor_list.keys()

with open(discussion_path, 'r') as f:
    for data in f:
        repo, _, items = data.split('\t')
        items = json.loads(items)
        for item in items:
            name = item['author']
            comments = item['comments']
            if name in contributor_list:
                if name not in contributor_pr:
                    contributor_pr[name] = 0
                contributor_pr[name] += 1
            for comment in comments:
                comment_name = comment['author']
                if comment_name in contributor_list:
                    if comment_name not in contributor_pr:
                        contributor_pr[comment_name] = 0
                    contributor_pr[comment_name] += 1
result = 0
for item in contributor_pr:
    if contributor_pr[item] > 0:
        result += 1
print(result)
    
discussion_with_contributor = 0
discussion_category = {}


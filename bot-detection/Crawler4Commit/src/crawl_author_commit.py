import json

import subprocess
import os
from tqdm import tqdm
import sys
import shutil

maxSize = 1024 * 1024 * 1024 * 10 # maxSize repo 10G
def folder_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def clone_and_parse_commit_history(repo_path,author):

    size = folder_size(repo_path) 
    if size > maxSize:
        return
    try:
        commits = subprocess.check_output(["git", "log","--author",author, "--name-status"], cwd = repo_path).decode("utf-8")

    except:
        print(repo_path)
    return commits

def clone_parse(reponame,authors):
    url = "https://github.com/" + reponame + ".git"
    try:
        subprocess.run(['git', 'clone', url], check=True)
        for author in authors:
            commits = clone_and_parse_commit_history(reponame.split('/')[-1],author)
            with open("../data/repo_user_commithistory.json", "a") as f:
                if commits:
                    answer = {"repo": reponame, "author": author, "commits": commits}
                    f.write(json.dumps(answer) + "\n")
        shutil.rmtree(reponame.split('/')[-1])
        return commits
    except:
        print(url)

repo_contributions = []
with open('../data/repo_contributions_flitered.json', 'r') as f:
    for line in f:
        repo_contributions.append(json.loads(line))
for repo_contributions in tqdm(repo_contributions[396:]):
    reponame = repo_contributions['repo']
    authors = repo_contributions['users']
    clone_parse(reponame,authors)


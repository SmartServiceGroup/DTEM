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


def clone_and_parse_commit_history(repo_path):

    size = folder_size(repo_path) 
    if size > maxSize:
        return
    commit_libraries = list()
    try:
        commits = subprocess.check_output(["git", "log", "--name-status"], cwd = repo_path).decode("utf-8")
        

    except:
        print(repo_path)
    return commits

def clone_parse(reponame):
    url = "https://github.com/" + reponame + ".git"
    try:
        subprocess.run(['git', 'clone', url], check=True)
        commits = clone_and_parse_commit_history(reponame.split('/')[-1])
        shutil.rmtree(reponame.split('/')[-1])
        return commits
    except:
        print(url)

with open('repos2crawlTopics50.json', 'r') as f:
    repos = json.load(f)
with open('../Repos-Info.json', 'r') as f:
    repos_info = json.load(f)
repo_commits = {}
for repoid in tqdm(repos[1700:]):
    reponame = repos_info[str(repoid)]['full_name']
    commits = clone_parse(reponame)
    if commits:
        repo_commits[repoid] = commits
    if repoid % 50 == 0:
        with open('repo_commits_T502.json', 'w') as f:
            json.dump(repo_commits, f)
with open('repo_commits_T502.json', 'w') as f:
    json.dump(repo_commits, f)
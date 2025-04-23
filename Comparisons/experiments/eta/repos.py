#!/usr/bin/env python3

from github import Github

def repo_contribution(repo_name, access_token=None):
    # Initialize the Github instance
    if access_token:
        g = Github(access_token)
    else:
        g = Github()
    
    try:
        # Get the repository object
        repo = g.get_repo(repo_name)
        
        # Initialize an empty list to store contributors and their contributions
        contributors = []
        
        # Get the contributors of the repository
        for contributor in repo.get_contributors():
            contributors.append((contributor.login, contributor.contributions))

        # Sort the list of contributors by contributions in descending order
        contributors.sort(key=lambda x: x[1], reverse=True)
        
        return contributors
    
    except Exception as e:
        # Handle any exceptions that occur during the request
        print(f"An error occurred: {e}")
        return []


# %% 

# read target_repos from target_repos.txt

target_repos = []
with open("./target_repos_names.txt", "r", encoding="utf-8") as inf:
    for line in inf:
        target_repos.append(line.strip())


# %% 

import pickle 
from tqdm import tqdm

# open and read result.pkl to data 
with open("./result.pkl", "rb") as inf:
    data = pickle.load(inf)

try: 
    for repo_name in tqdm(target_repos): 
        if repo_name in data:
            continue
        result = repo_contribution(repo_name, access_token='<sensored>')
        data[repo_name] = result
finally:
    with open("./result.pkl", "wb") as outf:
        pickle.dump(data, outf)

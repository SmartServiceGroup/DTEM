# nohup python Crawling_commit.py 0 0 275000 0 > logs/commit/0.log &

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
from shutil import copyfile
import time
import datetime
import os
import sys
import copy
from sys import argv
token, bg, ed, fn = [int(it) for it in argv[1:]]

with open('Ghtokens.txt', 'r') as ipfile:
    tokens = ipfile.readlines()
headers = {
    "Authorization" : 'token ' + tokens[token][:-1]
}
with open('repo_commits2crawl.json', 'r') as ipfile:
    repos_user = json.load(ipfile)
commits = {}
commits_s = set()
outputfile = 'Crawled/repo_commits/{}.json'.format(fn)
if os.path.exists(outputfile):
    with open(outputfile, 'r') as ipfile:
        iptext = ipfile.read()
        iptext = iptext[:-2] + '}'
        commits_s = json.loads(iptext)
    commits_s = set(commits_s.keys())
else:
    with open(outputfile, 'w') as opfile:
        opfile.write('{\n')
# print(commits_s)
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.05, status_forcelist=[ 443, 500, 502, 503, 504 ])
session.mount('https://', HTTPAdapter(max_retries=retries))
repos = list(repos_user.keys())
for i in range(bg, ed):
    repo = repos[i]
    if repo in commits_s:
        continue
    for user in repos_user[repo].keys():
        user_commits = []
        for page in range(700):
            url = 'https://api.github.com/repos/{}/commits?author={}&per_page=100&page={}'.format(repo, user, page)
            try:
                # strhtml = requests.get(url, headers=headers).text
                response = session.get(url, headers=headers)                                
                strhtml = response.text
            except Exception as e:
                print('28 Error:', url, e)
                break
            try:
                strhtml_json = json.loads(strhtml)
            except:
                is_json = 0
                if 'We had issues producing the response to your request' in strhtml:
                    cragain = 0
                    while cragain < 3:
                        cragain += 1
                        time.sleep(1)
                        # strhtml = requests.get(url, headers=headers).text
                        response = session.get(url, headers=headers)                                
                        strhtml = response.text
                        try:
                            strhtml_json = json.loads(strhtml)
                            is_json = 1
                            break
                        except:
                            if 'We had issues producing the response to your request' in strhtml:
                                continue
                            else:
                                break
                if not is_json:
                    print('33 Error:', url, strhtml)
                    break
            if not isinstance(strhtml_json, list):
                print('55 Error: not list', url, strhtml)
                
                if 'message' in strhtml_json.keys():
                    if strhtml_json["message"] in {"Not Found", "Repository access blocked", 'This repository is empty.'}:
                        print('41 Error:', i, url, strhtml_json["message"])
                        break
                    elif strhtml_json["message"].startswith("API rate limit exceeded"):
                        flt = 1
                        while isinstance(strhtml_json, dict) and 'message' in strhtml_json.keys() and strhtml_json["message"].startswith("API rate limit exceeded"):
                            now_time = datetime.datetime.now()
                            print(i, now_time.strftime('%H:%M:%S'))
                            sys.stdout.flush()
                            time.sleep(300)
                            # strhtml = requests.get(url, headers=headers).text
                            response = session.get(url, headers=headers)                                
                            strhtml = response.text
                            try:
                                strhtml_json = json.loads(strhtml)
                                flt = 0
                            except:
                                print('71 Error:', i, url, strhtml)
                                break
                        if flt:
                            continue
                    else:
                        print('76', i, url, strhtml)
                        continue
            
            for commit in strhtml_json:
                try:
                    new_commit = {'sha': commit['sha'], 'node_id': commit['node_id'], 'author_date': commit['commit']['author']['date'], 'committer_date': commit['commit']['committer']['date'], 'message': commit['commit']['message']}
                except:
                    print(strhtml_json)
                    exit(0)
                user_commits.append(new_commit)
            if len(strhtml_json) < 100:
                break
        if len(user_commits) > 0:
            if not repo in commits.keys():
                commits[repo] = {}
            commits[repo][user] = user_commits
            commits_s.add(repo)
    if i % 20 == 0:
        with open(outputfile, 'a') as opfile:
            for k, v in commits.items():
                opfile.write('"'+k + '":' + json.dumps(v, indent=1) + ',\n')
            # opfile.write(json.dumps(commits, indent=1))
            # opfile.write('\n')
        commits = {}
with open(outputfile, 'a') as opfile:
    for k, v in commits.items():
        opfile.write('"'+k + '":' + json.dumps(v, indent=1) + ',\n')
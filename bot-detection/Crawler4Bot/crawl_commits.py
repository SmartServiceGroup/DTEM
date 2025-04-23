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
    tokens = [it.split(' #')[0] for it in tokens]
    # print(tokens)
    # exit(0)
headers = {
    "Authorization" : 'token ' + tokens[token]
}

with open('Crawled/repo_commits/01.json', 'r') as ipfile:
    repo_commits = json.load(ipfile)
commits = {}
outputfile = 'Crawled/commits/{}.json'.format(fn)
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.05, status_forcelist=[ 443, 500, 502, 503, 504 ])
session.mount('https://', HTTPAdapter(max_retries=retries))
commit_files = {}
commits_s = set()
if os.path.exists(outputfile):
    with open(outputfile, 'r') as ipfile:
        iptext = ipfile.read()
        if len(iptext) > 2:
            
            iptext = iptext[:-2] + '}'
            commits_s = json.loads(iptext)
            commits_s = set(commits_s.keys())
else:
    with open(outputfile, 'w') as opfile:
        opfile.write('{\n')
repos = list(repo_commits.keys())
for i in range(bg, ed):
    repo = repos[i]
    if repo in commits_s:
        continue
    commit_files[repo] = {}
    commits_s.add(repo)
    for user, commits in repo_commits[repo].items():
        user_commits = {}
        for commit in commits:
            sha = commit['sha']
            url = 'https://api.github.com/repos/{}/commits/{}'.format(repo, sha)
            
            try:
                # strhtml = requests.get(url, headers=headers).text
                response = session.get(url, headers=headers)                                
                strhtml = response.text
            except Exception as e:
                print('28 Error:', url, e)
                continue
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
                    continue
            if not isinstance(strhtml_json, dict):
                print('55 Error: not dict', url, strhtml)
                continue
            if 'message' in strhtml_json.keys():
                if strhtml_json["message"] in {"Not Found", "Repository access blocked", 'This repository is empty.'}:
                    print('41 Error:', i, url, strhtml_json["message"])
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
            if not 'commit' in strhtml_json.keys():
                print('Error:', i, url, strhtml)
                continue
            files = strhtml_json['files']
            cfiles = []
            for file in files:
                cfiles.append({'filename': file['filename'], 'status': file['status']})
            if cfiles:
                user_commits[sha] = cfiles
        commit_files[repo][user] = user_commits
    
    if i % 10 == 0:
        with open(outputfile, 'a') as opfile:
            for k, v in commit_files.items():
                opfile.write('"'+k + '":' + json.dumps(v, indent=1) + ',\n')
            # opfile.write(json.dumps(commits, indent=1))
            # opfile.write('\n')
        commit_files = {}
with open(outputfile, 'a') as opfile:
    for k, v in commit_files.items():
        opfile.write('"'+k + '":' + json.dumps(v, indent=1) + ',\n')
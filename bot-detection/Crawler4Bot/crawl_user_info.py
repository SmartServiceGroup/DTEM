
import requests        
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
key_dict = {'per_page' : 100, 'state': 'all'}
with open('addedusers200_random.json', 'r') as ipfile:
    users = json.load(ipfile)
    
user_info = {}
if not os.path.exists('Crawled/user_info/{}.json'.format(fn)):
    with open('Crawled/user_info/{}.json'.format(fn), 'w') as opfile:
        opfile.write('{\n')
else:
    with open('Crawled/user_info/{}.json'.format(fn), 'r') as opfile:
        strhtml = opfile.read()
        strhtml = strhtml[:-2] + '}'
        user_info = json.loads(strhtml)
repoprojects = {}
write_user_info = {}
user_info = set(user_info.keys())
for i in range(bg, ed):
    user = users[i]
    if user in user_info:
        continue
    url = 'https://api.github.com/users/{}'.format(user)
    strhtml = requests.get(url, headers=headers).text
    strhtml_json = json.loads(strhtml)
    if 'message' in strhtml_json.keys():
        if strhtml_json["message"].startswith("Not Found"):
            continue
        if strhtml_json["message"].startswith("API rate limit exceeded"):
            while isinstance(strhtml_json, dict) and 'message' in strhtml_json.keys() and strhtml_json["message"].startswith("API rate limit exceeded"):
                now_time = datetime.datetime.now()
                print(i, now_time.strftime('%H:%M:%S'))
                sys.stdout.flush()
                time.sleep(300)
                strhtml = requests.get(url, params=key_dict, headers=headers).text
                try:
                    strhtml_json = json.loads(strhtml)
                    flt = 0
                except:
                    print('50 Error:', i, url, strhtml)
                    break
    if not 'name' in strhtml_json.keys() or not 'email' in strhtml_json.keys():
        print('Error:', i, url, strhtml)
        continue
    user_info.add(user)
    write_user_info[user] = {'id':strhtml_json['id'], 'node_id':strhtml_json['node_id'], 'name': strhtml_json['name'], 'email': strhtml_json['email']}
    if i % 20 == 0:
        with open('Crawled/user_info/' + str(fn) + '.json', 'a') as opfile:
            for user in write_user_info:
                strhtml = json.dumps(write_user_info[user], indent=1)
                opfile.write('"' + user + '":' + strhtml + ',\n')
        write_user_info = {}
with open('Crawled/user_info/' + str(fn) + '.json', 'a') as opfile:
    for user in write_user_info:
        strhtml = json.dumps(write_user_info[user], indent=1)
        opfile.write('"' + user + '":' + strhtml + ',\n')
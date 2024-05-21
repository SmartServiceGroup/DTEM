RecommendationTasks = [
    'ContributionRepo',
    'PRReviewer',
    'RepoMaintainer',
    'SimDeveloper'
]


PREFIX = {
    'ContributionRepo': './ContributionRepo/',
    'PRReviewer':'./PRReviewer/',
    'RepoMaintainer':'./RepoMaintainer/',
    'SimDeveloper':'./SimDeveloper/'
}

SUFFIX = {
    'ContributionRepo' : [
        'CF',
        'DTEM',
        'GAT',
        'Topic',
    ],
    'PRReviewer' : [
        'CF',
        'DTEM',
        'GAT'
    ],
    'RepoMaintainer' : [
        'CF',
        'DTEM',
        'GAT',
        'Topic',
    ],
    'SimDeveloper' : [
        'CF',
        'Dev2Vec',
        'DTEM',
        'GAT',
        'Topic',
    ]
}

SELECTED_DATA = {
    'ContributionRepo': './selected/ContributionRepo.json',
    'PRReviewer':'./selected/PRReviewer.json',
    'RepoMaintainer':'./selected/RepoMaintainer.json',
    'SimDeveloper':'./selected/SimDeveloper.json'
}

CONTRIBUTOR =   '../GNN/DataPreprocess/full_graph/content/contributors.json'
REPOSITORY =    '../GNN/DataPreprocess/full_graph/content/repositories.json'
PRS =           '../GNN/DataPreprocess/full_graph/content/prs.json'

RESULT_PREFIX = './result/result'

'''
import json 

for task in RecommendationTasks:
    for file in SUFFIX[task]:
        filename = PREFIX[task] + file + '.json'
        
        with open(filename, 'r') as f:
            data = json.load(f)
            print(filename, len(data))

'''
'''
./ContributionRepo/CF.json 42
./ContributionRepo/DTEM.json 42
./ContributionRepo/GAT.json 42
./ContributionRepo/Topic.json 42
./PRReviewer/CF.json 777
./PRReviewer/DTEM.json 777
./PRReviewer/GAT.json 853
./RepoMaintainer/CF.json 35
./RepoMaintainer/DTEM.json 35
./RepoMaintainer/GAT.json 35
./RepoMaintainer/Topic.json 35
./SimDeveloper/CF.json 1303
./SimDeveloper/Dev2Vec.json 3494
./SimDeveloper/DTEM.json 2799
./SimDeveloper/GAT.json 4216
./SimDeveloper/Topic.json 2799
'''
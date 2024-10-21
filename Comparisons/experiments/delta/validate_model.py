import json
import sys
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import numpy as np

from train_nn_10fold import Net, load_embeddings, collate_fn


class ContributionRepoDataset(Dataset):
    def __init__(self, contributor_idx, repo_idxs, repo_embedding, contributor_embedding, is_tensor=True) -> None:
        super().__init__()
        self.data = []
        src_embedding = contributor_embedding[contributor_idx]
        if is_tensor:
            src_embedding = src_embedding.cpu().numpy().tolist()
        for repo_idx in repo_idxs:
            dst_embedding = repo_embedding[repo_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().numpy().tolist()
            self.data.append([src_embedding + dst_embedding, repo_idx])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class PRReviewerDataset(Dataset):
    def __init__(self, repo_idx, pr_idx, contributor_idxs, repo_embedding, pr_embedding, contributor_embedding, is_tensor=True) -> None:
        super().__init__()
        self.data = []
        src_embedding = repo_embedding[repo_idx]
        if is_tensor:
            src_embedding = src_embedding.cpu().numpy().tolist()
        mid_embedding = pr_embedding[pr_idx]
        if is_tensor:
            mid_embedding = mid_embedding.cpu().numpy().tolist()
        for contributor_idx in contributor_idxs:
            dst_embedding = contributor_embedding[contributor_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().numpy().tolist()
            self.data.append([src_embedding + mid_embedding + dst_embedding, contributor_idx])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class RepoMaintainerDataset(Dataset):
    def __init__(self, repo_idx, contributor_idxs, repo_embedding, contributor_embedding, is_tensor=True) -> None:
        super().__init__()
        self.data = []
        src_embedding = repo_embedding[repo_idx]
        if is_tensor:
            src_embedding = src_embedding.cpu().numpy().tolist()
        for contributor_idx in contributor_idxs:
            dst_embedding = contributor_embedding[contributor_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().numpy().tolist()
            self.data.append([src_embedding + dst_embedding, contributor_idx])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SimDeveloperDataset(Dataset):
    def __init__(self, contributor_idx, c_idxs, contributor_embedding, is_tensor=True) -> None:
        super().__init__()
        self.data = []
        src_embedding = contributor_embedding[contributor_idx]
        if is_tensor:
            src_embedding = src_embedding.cpu().numpy().tolist()
        for c_idx in c_idxs:
            dst_embedding = contributor_embedding[c_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().numpy().tolist()
            self.data.append([src_embedding + dst_embedding, c_idx])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

task_configs = {
    'ContributionRepo': {
        'task': 'ContributionRepo', 
        'mp': 'ContributionRepo_result_ContributionRepo_mp.txt_09.bin',
        'wo_mp': 'ContributionRepo_result_ContributionRepo_wo_mp.txt_09.bin',
        'wo_mp_PCA': 'PCA/ContributionRepo_result_ContributionRepo_wo_mp_PCA.txt_08.bin',
        'valid_dataset_test': 'dataset_valid_test.json',
        'entity_names': ['contributor', 'repository'], 
    }, 
    'PRReviewer': {
        'task': 'PRReviewer', 
        'mp': 'PRReviewer_result_PRReviewer_mp.txt_09.bin',
        'wo_mp': 'PRReviewer_result_PRReviewer_wo_mp.txt_09.bin',
        'wo_mp_PCA': 'PCA/PRReviewer_result_PRReviewer_wo_mp_PCA.txt_09.bin',
        'valid_dataset_test': 'dataset_valid_test_modified.json',
        'entity_names': ['repository', 'pr', 'contributor']
    }, 
    'RepoMaintainer': {
        'task': 'RepoMaintainer', 
        'mp': 'RepoMaintainer_result_RepoMaintainer_mp.txt_09.bin',
        'wo_mp': 'RepoMaintainer_result_RepoMaintainer_wo_mp.txt_09.bin',
        'wo_mp_PCA': 'PCA/RepoMaintainer_result_RepoMaintainer_wo_mp_PCA.txt_09.bin',
        'valid_dataset_test':  'dataset_valid_test.json',
        'entity_names': ['repository', 'contributor']
    }, 
    'SimDeveloper': {
        'task': 'SimDeveloper', 
        'mp': 'SimDeveloper_result_SimDeveloper_mp.txt_09.bin',
        'wo_mp': 'SimDeveloper_result_SimDeveloper_wo_mp.txt_09.bin',
        'wo_mp_PCA': 'PCA/SimDeveloper_result_SimDeveloper_wo_mp_PCA.txt_09.bin',
        'valid_dataset_test': 'dataset_valid_test_modified.json',
        'entity_names': ['contributor', 'contributor']
    },
}

# for example:
# python SimDeveloper mp ../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_with_metapath.bin

def main(): 
    EXPECTED_ARGV_COUNT = 3
    if len(sys.argv) < EXPECTED_ARGV_COUNT: 
        print('Not enough arguement count')
        exit(1)

    task_name = sys.argv[1]
    model_suffix = sys.argv[2]
    emb_file_path = sys.argv[3]

    if task_name not in task_configs: 
        print(f'No such task: "{task_name}"', file=sys.stderr)
        exit(1)

    config = task_configs[task_name]

    # STEP 0: Get embeddings and load model
    embeddings = load_embeddings(emb_file_path, config['entity_names'])
    embedding_dim = sum([it.shape[1] for it in embeddings])

    with open(f'metric/{task_name}/{config["valid_dataset_test"]}', "r", encoding="utf-8") as inf:
        dataset = json.load(inf)

    model = Net(embedding_dim)
    model_path = 'bin/' + config[model_suffix]
    model.load_state_dict(torch.load(model_path, map_location='cpu'))


    # STEP 1. Now let's do some validating works.
    topks = {}
    # FIXME ugly appoarch... but it works. 
    for item in tqdm(dataset, total=len(dataset)):
        if task_name == 'ContributionRepo': 
            contributor_idx, search_scope, labels = item
            if len(labels) < 5:  
                continue
            d = ContributionRepoDataset(contributor_idx, search_scope, repo_embedding=embeddings[1], contributor_embedding=embeddings[0])
        elif task_name == 'PRReviewer': 
            repo_idx, pr_idx, search_scope, labels = item
            if len(search_scope) < 10:
                continue
            d = PRReviewerDataset(repo_idx, pr_idx, search_scope, embeddings[0], embeddings[1], embeddings[2])
        elif task_name == 'RepoMaintainer': 
            repo_idx, search_scope, labels = item
            if len(search_scope) < 10:
                continue
            d = RepoMaintainerDataset(repo_idx, search_scope, repo_embedding=embeddings[0], contributor_embedding=embeddings[1])
        elif task_name == 'SimDeveloper': 
            contributor_idx, search_scope, labels = item
            if len(labels) < 5:
                continue
            d = SimDeveloperDataset(contributor_idx, search_scope, embeddings[0])

        dataloader = DataLoader(d, batch_size=128, shuffle=False, collate_fn=collate_fn)
            
        model.eval()
        output = {}
        with torch.no_grad():
            for batch in dataloader:
                samples, contributor_idxs = batch
                results = model(samples).squeeze().numpy()
                if len(results.shape) == 0:
                    results = [results]
                for idx, result in zip(contributor_idxs, results):
                    output[idx] = result
        
        output = sorted(output.items(), key=lambda x: x[1], reverse=True)
        output = [x[0] for x in output[:20]]
        # print(output)
        rets = np.zeros(21).tolist()
        for i in range(1, 21):
            rets[i] = rets[i - 1]
            if i-1 < len(output) and output[i - 1] in labels:
                rets[i] = rets[i - 1] + 1


        key = \
            contributor_idx if task_name == 'ContributionRepo'  else \
            repo_idx        if task_name == 'PRReviewer'        else \
            repo_idx        if task_name == 'RepoMaintainer'    else \
            contributor_idx if task_name == 'SimDeveloper'      else \
            '<ERR>'
        topks[key] = (rets, search_scope)
    
    with open(f'metric_{task_name}_{model_suffix}.txt', "w", encoding="utf-8") as ouf:
        json.dump(topks, ouf, indent=4)

    return 0 

if __name__ == "__main__":
    main()
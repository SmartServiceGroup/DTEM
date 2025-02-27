import json
import sys
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import numpy as np

from train_nn_10fold import Net, load_embeddings, collate_fn

from colorama import Fore
from embedding_net import TASK_EMBEDDING_MAP

HL = Fore.YELLOW 
RST = Fore.RESET

class ContributionRepoDataset(Dataset):
    def __init__(self, contributor_idx, repo_idxs, repo_embedding, contributor_embedding, is_tensor=True) -> None:
        super().__init__()
        self.data = []
        src_embedding = contributor_embedding[contributor_idx]
        if is_tensor:
            src_embedding = src_embedding.cpu().detach().numpy().tolist()
        for repo_idx in repo_idxs:
            dst_embedding = repo_embedding[repo_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().detach().numpy().tolist()
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
            src_embedding = src_embedding.cpu().detach().numpy().tolist()
        mid_embedding = pr_embedding[pr_idx]
        if is_tensor:
            mid_embedding = mid_embedding.cpu().detach().numpy().tolist()
        for contributor_idx in contributor_idxs:
            dst_embedding = contributor_embedding[contributor_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().detach().numpy().tolist()
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
            src_embedding = src_embedding.cpu().detach().numpy().tolist()
        for contributor_idx in contributor_idxs:
            dst_embedding = contributor_embedding[contributor_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().detach().numpy().tolist()
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
            src_embedding = src_embedding.cpu().detach().numpy().tolist()
        for c_idx in c_idxs:
            dst_embedding = contributor_embedding[c_idx]
            if is_tensor:
                dst_embedding = dst_embedding.cpu().detach().numpy().tolist()
            self.data.append([src_embedding + dst_embedding, c_idx])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def main(
        task_name, 
        model_flp,
        embeddings_flp, 
        valid_dataset_test_flp,
        result_flp,
): 
    os.makedirs(os.path.dirname(result_flp), exist_ok=True)
    
    # STEP 0: Get embeddings and load model
    entity_names = TASK_EMBEDDING_MAP[task_name]
    embeddings = pickle.load(open(embeddings_flp, 'rb'))
    embeddings = [embeddings[entity_name] for entity_name in entity_names]

    embedding_dim = sum([it.shape[1] for it in embeddings])

    with open(valid_dataset_test_flp, 'r', encoding='utf-8') as inf:
        dataset = json.load(inf)

    model = Net(embedding_dim)
    model.load_state_dict(torch.load(model_flp, map_location='cpu'))

    # choice = input(f'For task {HL}{task_name}, {"with mp" if with_mp else "without mp"}{RST}. Continue? [Y/n] ')
    # if choice.upper() != 'Y': 
    #     return 0 

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
    
    with open(result_flp, "w", encoding="utf-8") as ouf:
        json.dump(topks, ouf, indent=4)

    return 0 


CONFIG = {
    'emb_v1': { 
        'with_mp': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/shrinked/emb_v1/ContributionRepo/model.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v1/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v1/HR/ContributionRepo.txt',
            }, 
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/shrinked/emb_v1/PRReviewer/model.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v1/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v1/HR/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/shrinked/emb_v1/RepoMaintainer/model.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v1/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v1/HR/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/shrinked/emb_v1/SimDeveloper/model.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v1/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v1/HR/SimDeveloper.txt',
            },
        }, 
        'without_mp': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/shrinked/emb_v1/ContributionRepo/model_wo_mp.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v1/entity_features_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v1/HR/ContributionRepo_wo_mp.txt',
            },
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/shrinked/emb_v1/PRReviewer/model_wo_mp.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v1/entity_features_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v1/HR/PRReviewer_wo_mp.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/shrinked/emb_v1/RepoMaintainer/model_wo_mp.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v1/entity_features_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v1/HR/RepoMaintainer_wo_mp.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/shrinked/emb_v1/SimDeveloper/model_wo_mp.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v1/entity_features_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v1/HR/SimDeveloper_wo_mp.txt',
            },
        }
    },
    'emb_v2': {
        'with_mp': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/shrinked/emb_v2/ContributionRepo/model.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v2/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v2/HR/ContributionRepo.txt',
            }, 
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/shrinked/emb_v2/PRReviewer/model.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v2/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v2/HR/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/shrinked/emb_v2/RepoMaintainer/model.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v2/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v2/HR/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/shrinked/emb_v2/SimDeveloper/model.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v2/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v2/HR/SimDeveloper.txt',
            },
        }, 
        'without_mp': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/shrinked/emb_v2/ContributionRepo/model_wo_mp.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v2/entity_features_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v2/HR/ContributionRepo_wo_mp.txt',
            },
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/shrinked/emb_v2/PRReviewer/model_wo_mp.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v2/entity_features_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v2/HR/PRReviewer_wo_mp.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/shrinked/emb_v2/RepoMaintainer/model_wo_mp.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v2/entity_features_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v2/HR/RepoMaintainer_wo_mp.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/shrinked/emb_v2/SimDeveloper/model_wo_mp.bin.09',
                'embeddings_flp':           'data/shrinked_embeddings/emb_v2/entity_features_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/shrinked/emb_v2/HR/SimDeveloper_wo_mp.txt',
            },
        }
    }, 
    'emb_blended': {
        'our_method_contributor_gnn_only': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/blended/ContributionRepo/model_our_method.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/our_method_contributor_gnn_only/ContributionRepo.txt',
            }, 
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/blended/PRReviewer/model_our_method.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/our_method_contributor_gnn_only/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/blended/RepoMaintainer/model_our_method.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/our_method_contributor_gnn_only/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/blended/SimDeveloper/model_our_method.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/our_method_contributor_gnn_only/SimDeveloper.txt',
            },
        }, 
        'ablation_with_mp': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/blended/ContributionRepo/model_pca_mp.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features_pca_mp.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/ablation_with_mp/ContributionRepo.txt',
            }, 
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/blended/PRReviewer/model_pca_mp.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features_pca_mp.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/ablation_with_mp/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/blended/RepoMaintainer/model_pca_mp.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features_pca_mp.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/ablation_with_mp/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/blended/SimDeveloper/model_pca_mp.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features_pca_mp.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/ablation_with_mp/SimDeveloper.txt',
            },
        },
        'ablation_without_mp': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/blended/ContributionRepo/model_pca_wo_mp.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features_pca_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/ablation_without_mp/ContributionRepo.txt',
            },
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/blended/PRReviewer/model_pca_wo_mp.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features_pca_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/ablation_without_mp/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/blended/RepoMaintainer/model_pca_wo_mp.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features_pca_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/ablation_without_mp/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/blended/SimDeveloper/model_pca_wo_mp.bin.09',
                'embeddings_flp':           'data/blended_embeddings/entity_features_pca_wo_mp.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/blended/HR/ablation_without_mp/SimDeveloper.txt',
            },
        },
    },
    'emb_noise': {
        'noise_scale_0.5': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/noise/0.5/ContributionRepo/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.5_noise.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.5/ContributionRepo.txt',
            },
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/noise/0.5/PRReviewer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.5_noise.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.5/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/noise/0.5/RepoMaintainer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.5_noise.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.5/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/noise/0.5/SimDeveloper/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.5_noise.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.5/SimDeveloper.txt',
            },
        },
        'noise_scale_0.2': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/noise/0.2/ContributionRepo/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.2_noise.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.2/ContributionRepo.txt',
            },
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/noise/0.2/PRReviewer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.2_noise.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.2/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/noise/0.2/RepoMaintainer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.2_noise.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.2/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/noise/0.2/SimDeveloper/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.2_noise.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.2/SimDeveloper.txt',
            }
        }, 
        'noise_scale_0.7': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/noise/0.7/ContributionRepo/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.7_noise.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.7/ContributionRepo.txt',
            },
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/noise/0.7/PRReviewer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.7_noise.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.7/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/noise/0.7/RepoMaintainer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.7_noise.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.7/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/noise/0.7/SimDeveloper/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.7_noise.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.7/SimDeveloper.txt',
            }
        },
        'noise_scale_1.0': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/noise/1.0/ContributionRepo/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_1.0_noise.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/1.0/ContributionRepo.txt',
            },
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/noise/1.0/PRReviewer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_1.0_noise.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/1.0/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/noise/1.0/RepoMaintainer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_1.0_noise.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/1.0/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/noise/1.0/SimDeveloper/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_1.0_noise.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/1.0/SimDeveloper.txt',
            },
        },
        'noise_scale_0.3': { 
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/noise/0.3/ContributionRepo/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.3_noise.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.3/ContributionRepo.txt',
            },
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/noise/0.3/PRReviewer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.3_noise.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.3/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/noise/0.3/RepoMaintainer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.3_noise.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.3/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/noise/0.3/SimDeveloper/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.3_noise.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.3/SimDeveloper.txt',
            },
        },
        'noise_scale_0.4': {
            'ContributionRepo': {
                'task_name':                'ContributionRepo',
                'model_flp':                'bin/noise/0.4/ContributionRepo/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.4_noise.pkl',
                'valid_dataset_test_flp':   'metric/ContributionRepo/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.4/ContributionRepo.txt',
            },
            'PRReviewer': {
                'task_name':                'PRReviewer',
                'model_flp':                'bin/noise/0.4/PRReviewer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.4_noise.pkl',
                'valid_dataset_test_flp':   'metric/PRReviewer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.4/PRReviewer.txt',
            },
            'RepoMaintainer': {
                'task_name':                'RepoMaintainer',
                'model_flp':                'bin/noise/0.4/RepoMaintainer/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.4_noise.pkl',
                'valid_dataset_test_flp':   'metric/RepoMaintainer/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.4/RepoMaintainer.txt',
            },
            'SimDeveloper': {
                'task_name':                'SimDeveloper',
                'model_flp':                'bin/noise/0.4/SimDeveloper/model.bin.09',
                'embeddings_flp':           'data/noise_embeddings/entity_features_0.4_noise.pkl',
                'valid_dataset_test_flp':   'metric/SimDeveloper/dataset_valid_test.json',
                'result_flp':               'result/noise/HR/0.4/SimDeveloper.txt',
            },
        }
    }
}

if __name__ == "__main__":
    # [
    #     print(it['SimDeveloper']['result_flp'])
    #     for it in CONFIG['emb_noise'].values()
    # ]

    # main(**CONFIG['emb_noise']['noise_scale_0.7']['ContributionRepo'])
    main(**CONFIG['emb_noise']['noise_scale_0.7']['PRReviewer'])
    # main(**CONFIG['emb_noise']['noise_scale_0.7']['RepoMaintainer'])

    # main(**CONFIG['emb_noise']['noise_scale_1.0']['ContributionRepo'])
    main(**CONFIG['emb_noise']['noise_scale_1.0']['PRReviewer'])
    # main(**CONFIG['emb_noise']['noise_scale_1.0']['RepoMaintainer'])

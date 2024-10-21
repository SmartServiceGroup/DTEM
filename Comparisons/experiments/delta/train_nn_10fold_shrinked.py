'''
Copied from "RecommendationTasks/RepoMaintainer/train_nn_10fold.py", modified.
'''

import json
import os
import numpy
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from dgl import load_graphs

from embedding_net import TASK_EMBEDDING_MAP


class Net(nn.Module):
    def __init__(self, total_dim):
        super(Net, self).__init__()
        self.total_dim = total_dim 
        self.fc1 = nn.Linear(self.total_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.sigmoid(x).squeeze()


class MyDataset(Dataset):
    def __init__(self, samples, embedding_a, embedding_b) -> None:
        super().__init__()
        self.data = []
        if isinstance(samples, str):
            with open(samples, "r", encoding="utf-8") as inf:
                samples = json.load(inf)
        for sample in samples:
            assert len(sample) == 3
            entity_a_idx, pos_entity_b_idx, neg_entity_b_idx = sample
            self.data.append([
                torch.cat([embedding_a[entity_a_idx], embedding_b[pos_entity_b_idx]], dim=-1).detach().numpy().tolist(),
                1,
            ])
            self.data.append([
                torch.cat([embedding_a[entity_a_idx], embedding_b[neg_entity_b_idx]], dim=-1).detach().numpy().tolist(),
                0,
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MyDatasetModified(Dataset):
    def __init__(self, samples, repo_node_embedding, pr_embedding, contributor_embedding) -> None:
        super().__init__()
        self.data = []
        if isinstance(samples, str):
            with open(samples, "r", encoding="utf-8") as inf:
                samples = json.load(inf)
        for sample in samples:
            assert len(sample) == 4
            repo_idx, pr_idx, pos_reviewer_idx, neg_reviewer_idx = sample
            self.data.append([
                torch.cat([repo_node_embedding[repo_idx], pr_embedding[pr_idx], contributor_embedding[pos_reviewer_idx]], dim=-1).detach().numpy().tolist(),
                1,
            ])
            self.data.append([
                torch.cat([repo_node_embedding[repo_idx], pr_embedding[pr_idx], contributor_embedding[neg_reviewer_idx]], dim=-1).detach().numpy().tolist(),
                0,
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    samples = []
    labels = []
    for sample in batch:
        samples.append(sample[0])
        labels.append(sample[1])
    return torch.FloatTensor(samples), torch.LongTensor(labels)

def metric(pos_right, neg_right, pos_total, neg_total):
    EPS = 1e-5
    precision = (pos_right + EPS) / (pos_right + neg_total - neg_right + EPS)
    recall = (pos_right + EPS) / (pos_total + EPS)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def main(
        embeddings_flp,
        result_flp,
        task_name,
        model_flp,
        device='cuda:0',
):
    input(f'Now running task {task_name}. {result_flp = }. \nPress any key to continue...')

    os.makedirs(os.path.dirname(result_flp), exist_ok=True)
    os.makedirs(os.path.dirname(model_flp), exist_ok=True)
    
    entity_names = TASK_EMBEDDING_MAP[task_name]

    K = 10
    epochs = 60
    stat_precision = []
    stat_recall = []
    stat_f1 = []

    # Step 1. Create dataset
    embeddings = pickle.load(open(embeddings_flp, 'rb'))
    embeddings = [embeddings[entity_name] for entity_name in entity_names]

    # FIXME: Ugly approach! (ugly, but works)
    # Depends on the fact that expect task `PRReviewer`, 
    # all tasks just use two kinds of entities. 
    if task_name != 'PRReviewer': 
        emb_a, emb_b = embeddings
        embedding_dim = emb_a.shape[1] + emb_b.shape[1]   # total input dim
    else: 
        emb_repo, emb_pr, emb_cont = embeddings 
        embedding_dim = emb_repo.shape[1] + emb_pr.shape[1] + emb_cont.shape[1] 
    
    for i in range(K):
        train_sample_path = f'./data/{task_name}/10fold/train{i}.json'
        test_sample_path =  f'./data/{task_name}/10fold/test{i}.json'
        model_flp_local = model_flp + f'.{i:02d}'

        if task_name != 'PRReviewer': 
            train_dataset = MyDataset(samples=train_sample_path, embedding_a=emb_a, embedding_b=emb_b)
            test_dataset  = MyDataset(samples=test_sample_path, embedding_a=emb_a, embedding_b=emb_b)
        else: 
            train_dataset = MyDatasetModified(samples=train_sample_path, repo_node_embedding=emb_repo, pr_embedding=emb_pr, contributor_embedding=emb_cont)
            test_dataset  = MyDatasetModified(samples=test_sample_path,  repo_node_embedding=emb_repo, pr_embedding=emb_pr, contributor_embedding=emb_cont)
    
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_dataloader  = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

        model = Net(total_dim=embedding_dim)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.BCELoss()

        # Training loop
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            model.train()
            for batch_idx, (x, labels) in enumerate(train_dataloader):
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        torch.save(model.state_dict(), model_flp_local)
    
        # Test
        model.load_state_dict(torch.load(model_flp_local))
        model.eval()
        pos_rights = 0
        neg_rights = 0
        pos_totals = 0
        neg_totals = 0
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_dataloader):
                x, labels = x.to(device), labels.to(device)
                logits = model(x)
                pred = (logits > 0.5).long()
                pos_right = ((pred == 1) & (labels == 1)).sum().item()
                neg_right = ((pred == 0) & (labels == 0)).sum().item()
                pos_total = (labels == 1).sum().item()
                neg_total = (labels == 0).sum().item()

                pos_rights += pos_right
                neg_rights += neg_right
                pos_totals += pos_total
                neg_totals += neg_total
        precision, recall, f1 = metric(pos_rights, neg_rights, pos_totals, neg_totals)
        
        stat_precision.append(precision)
        stat_recall.append(recall)
        stat_f1.append(f1)
        print(f"Test: precision={precision}, recall={recall}, f1={f1}")

    avg_precision = numpy.average(stat_precision)
    avg_recall = numpy.average(stat_recall)
    avg_f1 = numpy.average(stat_f1)

    with open(result_flp, 'w') as f:
        f.write(str(stat_precision) + '\n')
        f.write(str(stat_recall) + '\n')
        f.write(str(stat_f1) + '\n')
        f.write("Average precision: " + str(avg_precision) + '\n')
        f.write("Average recall: " + str(avg_recall) + '\n')
        f.write("Average f1: " + str(avg_f1) + '\n')
        f.write(f'{avg_precision:.3f} | {avg_recall:.3f} | {avg_f1:.3f} |')


CONFIG = {
    'v1': {
        'emb_average': {
            'emb_v1': {
                'with_mp': {
                    'ContributionRepo': {
                        'task_name':        'ContributionRepo',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v1/entity_features.pkl',
                        'result_flp':       'result/shrinked/emb_v1/PRF1/ContributionRepo.txt',
                        'model_flp':        'bin/shrinked/emb_v1/ContributionRepo/model.bin',
                        'device':           'cuda:0',
                    },
                    'PRReviewer': {
                        'task_name':        'PRReviewer',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v1/entity_features.pkl',
                        'result_flp':       'result/shrinked/emb_v1/PRF1/PRReviewer.txt',
                        'model_flp':        'bin/shrinked/emb_v1/PRReviewer/model.bin',
                        'device':           'cuda:0',
                    }, 
                    'RepoMaintainer': {
                        'task_name':        'RepoMaintainer',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v1/entity_features.pkl',
                        'result_flp':       'result/shrinked/emb_v1/PRF1/RepoMaintainer.txt',
                        'model_flp':        'bin/shrinked/emb_v1/RepoMaintainer/model.bin',
                        'device':           'cuda:0',
                    },
                    'SimDeveloper': {
                        'task_name':        'SimDeveloper',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v1/entity_features.pkl',
                        'result_flp':       'result/shrinked/emb_v1/PRF1/SimDeveloper.txt',
                        'model_flp':        'bin/shrinked/emb_v1/SimDeveloper/model.bin',
                        'device':           'cuda:0',
                    },
                }, 
                'without_mp': {
                    'ContributionRepo': {
                        'task_name':        'ContributionRepo',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v1/entity_features_wo_mp.pkl',
                        'result_flp':       'result/shrinked/emb_v1/PRF1/ContributionRepo_wo_mp.txt',
                        'model_flp':        'bin/shrinked/emb_v1/ContributionRepo/model_wo_mp.bin',
                        'device':           'cuda:1',
                    },
                    'PRReviewer': {
                        'task_name':        'PRReviewer',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v1/entity_features_wo_mp.pkl',
                        'result_flp':       'result/shrinked/emb_v1/PRF1/PRReviewer_wo_mp.txt',
                        'model_flp':        'bin/shrinked/emb_v1/PRReviewer/model_wo_mp.bin',
                        'device':           'cuda:1',
                    },
                    'RepoMaintainer': {
                        'task_name':        'RepoMaintainer',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v1/entity_features_wo_mp.pkl',
                        'result_flp':       'result/shrinked/emb_v1/PRF1/RepoMaintainer_wo_mp.txt',
                        'model_flp':        'bin/shrinked/emb_v1/RepoMaintainer/model_wo_mp.bin',
                        'device':           'cuda:1',
                    },
                    'SimDeveloper': {
                        'task_name':        'SimDeveloper',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v1/entity_features_wo_mp.pkl',
                        'result_flp':       'result/shrinked/emb_v1/PRF1/SimDeveloper_wo_mp.txt',
                        'model_flp':        'bin/shrinked/emb_v1/SimDeveloper/model_wo_mp.bin',
                        'device':           'cuda:1',
                    },
                }
            }, 
            'emb_v2': {
                'with_mp': {
                    'ContributionRepo': {
                        'task_name':        'ContributionRepo',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v2/entity_features.pkl',
                        'result_flp':       'result/shrinked/emb_v2/PRF1/ContributionRepo.txt',
                        'model_flp':        'bin/shrinked/emb_v2/ContributionRepo/model.bin',
                        'device':           'cuda:0',
                    },
                    'PRReviewer': {
                        'task_name':        'PRReviewer',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v2/entity_features.pkl',
                        'result_flp':       'result/shrinked/emb_v2/PRF1/PRReviewer.txt',
                        'model_flp':        'bin/shrinked/emb_v2/PRReviewer/model.bin',
                        'device':           'cuda:0',
                    }, 
                    'RepoMaintainer': {
                        'task_name':        'RepoMaintainer',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v2/entity_features.pkl',
                        'result_flp':       'result/shrinked/emb_v2/PRF1/RepoMaintainer.txt',
                        'model_flp':        'bin/shrinked/emb_v2/RepoMaintainer/model.bin',
                        'device':           'cuda:0',
                    },
                    'SimDeveloper': {
                        'task_name':        'SimDeveloper',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v2/entity_features.pkl',
                        'result_flp':       'result/shrinked/emb_v2/PRF1/SimDeveloper.txt',
                        'model_flp':        'bin/shrinked/emb_v2/SimDeveloper/model.bin',
                        'device':           'cuda:0',
                    },
                }, 
                'without_mp': {
                    'ContributionRepo': {
                        'task_name':        'ContributionRepo',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v2/entity_features_wo_mp.pkl',
                        'result_flp':       'result/shrinked/emb_v2/PRF1/ContributionRepo_wo_mp.txt',
                        'model_flp':        'bin/shrinked/emb_v2/ContributionRepo/model_wo_mp.bin',
                        'device':           'cuda:0',
                    },
                    'PRReviewer': {
                        'task_name':        'PRReviewer',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v2/entity_features_wo_mp.pkl',
                        'result_flp':       'result/shrinked/emb_v2/PRF1/PRReviewer_wo_mp.txt',
                        'model_flp':        'bin/shrinked/emb_v2/PRReviewer/model_wo_mp.bin',
                        'device':           'cuda:0',
                    },
                    'RepoMaintainer': {
                        'task_name':        'RepoMaintainer',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v2/entity_features_wo_mp.pkl',
                        'result_flp':       'result/shrinked/emb_v2/PRF1/RepoMaintainer_wo_mp.txt',
                        'model_flp':        'bin/shrinked/emb_v2/RepoMaintainer/model_wo_mp.bin',
                        'device':           'cuda:0',
                    },
                    'SimDeveloper': {
                        'task_name':        'SimDeveloper',
                        'embeddings_flp':   'data/shrinked_embeddings/emb_v2/entity_features_wo_mp.pkl',
                        'result_flp':       'result/shrinked/emb_v2/PRF1/SimDeveloper_wo_mp.txt',
                        'model_flp':        'bin/shrinked/emb_v2/SimDeveloper/model_wo_mp.bin',
                        'device':           'cuda:0',
                    },
                }
            },
        },
        'emb_blended': {
            'our_method_contributor_gnn_only': {
                'ContributionRepo': {
                    'task_name':        'ContributionRepo',
                    'embeddings_flp':   'data/blended_embeddings/entity_features.pkl',
                    'result_flp':       'result/blended/PRF1/ContributionRepo_our_method.txt',
                    'model_flp':        'bin/blended/ContributionRepo/model_our_method.bin',
                    'device':           'cuda:0',
                },
                'PRReviewer': {
                    'task_name':        'PRReviewer',
                    'embeddings_flp':   'data/blended_embeddings/entity_features.pkl',
                    'result_flp':       'result/blended/PRF1/PRReviewer_our_method.txt',
                    'model_flp':        'bin/blended/PRReviewer/model_our_method.bin',
                    'device':           'cuda:0',
                },
                'RepoMaintainer': {
                    'task_name':        'RepoMaintainer',
                    'embeddings_flp':   'data/blended_embeddings/entity_features.pkl',
                    'result_flp':       'result/blended/PRF1/RepoMaintainer_our_method.txt',
                    'model_flp':        'bin/blended/RepoMaintainer/model_our_method.bin',
                    'device':           'cuda:0',
                },
                'SimDeveloper': {
                    'task_name':        'SimDeveloper',
                    'embeddings_flp':   'data/blended_embeddings/entity_features.pkl',
                    'result_flp':       'result/blended/PRF1/SimDeveloper_our_method.txt',
                    'model_flp':        'bin/blended/SimDeveloper/model_our_method.bin',
                    'device':           'cuda:0',
                }
            },
            'ablation_with_mp': {
                'ContributionRepo': {
                    'task_name':        'ContributionRepo',
                    'embeddings_flp':   'data/blended_embeddings/entity_features_pca_mp.pkl',
                    'result_flp':       'result/blended/PRF1/ContributionRepo_pca_mp.txt',
                    'model_flp':        'bin/blended/ContributionRepo/model_pca_mp.bin',
                    'device':           'cuda:0',
                },
                'PRReviewer': {
                    'task_name':        'PRReviewer',
                    'embeddings_flp':   'data/blended_embeddings/entity_features_pca_mp.pkl',
                    'result_flp':       'result/blended/PRF1/PRReviewer_pca_mp.txt',
                    'model_flp':        'bin/blended/PRReviewer/model_pca_mp.bin',
                    'device':           'cuda:0',
                },
                'RepoMaintainer': {
                    'task_name':        'RepoMaintainer',
                    'embeddings_flp':   'data/blended_embeddings/entity_features_pca_mp.pkl',
                    'result_flp':       'result/blended/PRF1/RepoMaintainer_pca_mp.txt',
                    'model_flp':        'bin/blended/RepoMaintainer/model_pca_mp.bin',
                    'device':           'cuda:0',
                },
                'SimDeveloper': {
                    'task_name':        'SimDeveloper',
                    'embeddings_flp':   'data/blended_embeddings/entity_features_pca_mp.pkl',
                    'result_flp':       'result/blended/PRF1/SimDeveloper_pca_mp.txt',
                    'model_flp':        'bin/blended/SimDeveloper/model_pca_mp.bin',
                    'device':           'cuda:0',
                }
            },
            'ablation_without_mp': {
                'ContributionRepo': {
                    'task_name':        'ContributionRepo',
                    'embeddings_flp':   'data/blended_embeddings/entity_features_pca_wo_mp.pkl',
                    'result_flp':       'result/blended/PRF1/ContributionRepo_pca_wo_mp.txt',
                    'model_flp':        'bin/blended/ContributionRepo/model_pca_wo_mp.bin',
                    'device':           'cuda:0',
                },
                'PRReviewer': {
                    'task_name':        'PRReviewer',
                    'embeddings_flp':   'data/blended_embeddings/entity_features_pca_wo_mp.pkl',
                    'result_flp':       'result/blended/PRF1/PRReviewer_pca_wo_mp.txt',
                    'model_flp':        'bin/blended/PRReviewer/model_pca_wo_mp.bin',
                    'device':           'cuda:0',
                },
                'RepoMaintainer': {
                    'task_name':        'RepoMaintainer',
                    'embeddings_flp':   'data/blended_embeddings/entity_features_pca_wo_mp.pkl',
                    'result_flp':       'result/blended/PRF1/RepoMaintainer_pca_wo_mp.txt',
                    'model_flp':        'bin/blended/RepoMaintainer/model_pca_wo_mp.bin',
                    'device':           'cuda:0',
                },
                'SimDeveloper': {
                    'task_name':        'SimDeveloper',
                    'embeddings_flp':   'data/blended_embeddings/entity_features_pca_wo_mp.pkl',
                    'result_flp':       'result/blended/PRF1/SimDeveloper_pca_wo_mp.txt',
                    'model_flp':        'bin/blended/SimDeveloper/model_pca_wo_mp.bin',
                    'device':           'cuda:0',
                },
            },
        },
        'emb_noise': {
            'noise_scale_0.5': {
                'ContributionRepo': {
                    'task_name':        'ContributionRepo',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.5_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.5/ContributionRepo.txt',
                    'model_flp':        'bin/noise/0.5/ContributionRepo/model.bin',
                    'device':           'cuda:0',
                },
                'PRReviewer': {
                    'task_name':        'PRReviewer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.5_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.5/PRReviewer.txt',
                    'model_flp':        'bin/noise/0.5/PRReviewer/model.bin',
                    'device':           'cuda:0',
                },
                'RepoMaintainer': {
                    'task_name':        'RepoMaintainer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.5_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.5/RepoMaintainer.txt',
                    'model_flp':        'bin/noise/0.5/RepoMaintainer/model.bin',
                    'device':           'cuda:0',
                },
                'SimDeveloper': {
                    'task_name':        'SimDeveloper',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.5_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.5/SimDeveloper.txt',
                    'model_flp':        'bin/noise/0.5/SimDeveloper/model.bin',
                    'device':           'cuda:0',
                },
            },
            'noise_scale_0.2': {
                'ContributionRepo': {
                    'task_name':        'ContributionRepo',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.2_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.2/ContributionRepo.txt',
                    'model_flp':        'bin/noise/0.2/ContributionRepo/model.bin',
                    'device':           'cuda:0',
                },
                'PRReviewer': {
                    'task_name':        'PRReviewer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.2_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.2/PRReviewer.txt',
                    'model_flp':        'bin/noise/0.2/PRReviewer/model.bin',
                    'device':           'cuda:0',
                },
                'RepoMaintainer': {
                    'task_name':        'RepoMaintainer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.2_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.2/RepoMaintainer.txt',
                    'model_flp':        'bin/noise/0.2/RepoMaintainer/model.bin',
                    'device':           'cuda:0',
                },
                'SimDeveloper': {
                    'task_name':        'SimDeveloper',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.2_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.2/SimDeveloper.txt',
                    'model_flp':        'bin/noise/0.2/SimDeveloper/model.bin',
                }
            },
            'noise_scale_0.7': {
                'ContributionRepo': {
                    'task_name':        'ContributionRepo',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.7_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.7/ContributionRepo.txt',
                    'model_flp':        'bin/noise/0.7/ContributionRepo/model.bin',
                },
                'PRReviewer': {
                    'task_name':        'PRReviewer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.7_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.7/PRReviewer.txt',
                    'model_flp':        'bin/noise/0.7/PRReviewer/model.bin',
                },
                'RepoMaintainer': {
                    'task_name':        'RepoMaintainer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.7_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.7/RepoMaintainer.txt',
                    'model_flp':        'bin/noise/0.7/RepoMaintainer/model.bin',
                },
                'SimDeveloper': {
                    'task_name':        'SimDeveloper',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.7_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.7/SimDeveloper.txt',
                    'model_flp':        'bin/noise/0.7/SimDeveloper/model.bin',
                },
            },
            'noise_scale_1.0': {
                'ContributionRepo': {
                    'task_name':        'ContributionRepo',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_1.0_noise.pkl',
                    'result_flp':       'result/noise/PRF1/1.0/ContributionRepo.txt',
                    'model_flp':        'bin/noise/1.0/ContributionRepo/model.bin',
                },
                'PRReviewer': {
                    'task_name':        'PRReviewer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_1.0_noise.pkl',
                    'result_flp':       'result/noise/PRF1/1.0/PRReviewer.txt',
                    'model_flp':        'bin/noise/1.0/PRReviewer/model.bin',
                },
                'RepoMaintainer': {
                    'task_name':        'RepoMaintainer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_1.0_noise.pkl',
                    'result_flp':       'result/noise/PRF1/1.0/RepoMaintainer.txt',
                    'model_flp':        'bin/noise/1.0/RepoMaintainer/model.bin',
                },
                'SimDeveloper': {
                    'task_name':        'SimDeveloper',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_1.0_noise.pkl',
                    'result_flp':       'result/noise/PRF1/1.0/SimDeveloper.txt',
                    'model_flp':        'bin/noise/1.0/SimDeveloper/model.bin',
                },
            }, 
            'noise_scale_0.3': {
                'ContributionRepo': {
                    'task_name':        'ContributionRepo',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.3_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.3/ContributionRepo.txt',
                    'model_flp':        'bin/noise/0.3/ContributionRepo/model.bin',
                },
                'PRReviewer': {
                    'task_name':        'PRReviewer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.3_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.3/PRReviewer.txt',
                    'model_flp':        'bin/noise/0.3/PRReviewer/model.bin',
                },
                'RepoMaintainer': {
                    'task_name':        'RepoMaintainer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.3_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.3/RepoMaintainer.txt',
                    'model_flp':        'bin/noise/0.3/RepoMaintainer/model.bin',
                },
                'SimDeveloper': {
                    'task_name':        'SimDeveloper',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.3_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.3/SimDeveloper.txt',
                    'model_flp':        'bin/noise/0.3/SimDeveloper/model.bin',
                },
            },
            'noise_scale_0.4': {
                'ContributionRepo': {
                    'task_name':        'ContributionRepo',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.4_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.4/ContributionRepo.txt',
                    'model_flp':        'bin/noise/0.4/ContributionRepo/model.bin',
                },
                'PRReviewer': {
                    'task_name':        'PRReviewer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.4_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.4/PRReviewer.txt',
                    'model_flp':        'bin/noise/0.4/PRReviewer/model.bin',
                },
                'RepoMaintainer': {
                    'task_name':        'RepoMaintainer',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.4_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.4/RepoMaintainer.txt',
                    'model_flp':        'bin/noise/0.4/RepoMaintainer/model.bin',
                },
                'SimDeveloper': {
                    'task_name':        'SimDeveloper',
                    'embeddings_flp':   'data/noise_embeddings/entity_features_0.4_noise.pkl',
                    'result_flp':       'result/noise/PRF1/0.4/SimDeveloper.txt',
                    'model_flp':        'bin/noise/0.4/SimDeveloper/model.bin',
                },
            },
        },
    },
}


if __name__ == "__main__":
    print('Don\'t run this now!')
    exit(1)

    # main(**CONFIG['v1']['emb_noise']['noise_scale_0.7']['ContributionRepo'])
    # main(**CONFIG['v1']['emb_noise']['noise_scale_0.7']['PRReviewer'])
    # main(**CONFIG['v1']['emb_noise']['noise_scale_0.7']['RepoMaintainer'])

    # main(**CONFIG['v1']['emb_noise']['noise_scale_1.0']['ContributionRepo'])
    # main(**CONFIG['v1']['emb_noise']['noise_scale_1.0']['PRReviewer'])
    # main(**CONFIG['v1']['emb_noise']['noise_scale_1.0']['RepoMaintainer'])

#  -------------------------------------------------------------------------------- 
    import sys
    [
        lambda: main(**CONFIG['v1']['emb_noise']['noise_scale_0.3']['ContributionRepo']),
        lambda: main(**CONFIG['v1']['emb_noise']['noise_scale_0.3']['PRReviewer']),
        lambda: main(**CONFIG['v1']['emb_noise']['noise_scale_0.3']['RepoMaintainer']),
        lambda: main(**CONFIG['v1']['emb_noise']['noise_scale_0.3']['SimDeveloper']),
        lambda: main(**CONFIG['v1']['emb_noise']['noise_scale_0.4']['ContributionRepo']),
        lambda: main(**CONFIG['v1']['emb_noise']['noise_scale_0.4']['PRReviewer']),
        lambda: main(**CONFIG['v1']['emb_noise']['noise_scale_0.4']['RepoMaintainer']),
        lambda: main(**CONFIG['v1']['emb_noise']['noise_scale_0.4']['SimDeveloper']),
    ][int(sys.argv[1])]()
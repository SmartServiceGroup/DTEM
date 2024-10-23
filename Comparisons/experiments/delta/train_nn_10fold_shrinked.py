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

    emb_a, emb_b = embeddings
    embedding_dim = emb_a.shape[1] + emb_b.shape[1]   # total input dim
    
    for i in range(K):
        train_sample_path = f'./data/{task_name}/10fold/train{i}.json'
        test_sample_path =  f'./data/{task_name}/10fold/test{i}.json'
        model_flp_local = model_flp + f'.{i:02d}'

        train_dataset = MyDataset(samples=train_sample_path, embedding_a=emb_a, embedding_b=emb_b)
        test_dataset  = MyDataset(samples=test_sample_path, embedding_a=emb_a, embedding_b=emb_b)
    
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
        },
    },
}


if __name__ == "__main__":
    main(**CONFIG['v1']['emb_average']['emb_v1']['with_mp']['ContributionRepo'])
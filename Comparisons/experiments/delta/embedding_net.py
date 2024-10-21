#!/usr/bin/env python3

import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import *
import json
import random
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from dgl import load_graphs

random.seed(42)

ALL_TASKS = ['ContributionRepo', 'PRReviewer', 'RepoMaintainer', 'SimDeveloper']
ALL_ENTITIES = ['contributor', 'repository', 'pr', 'issue']

# See the table in `if __name__ == '__main__'` in 
# Comparisons/experiments/delta/train_nn_10fold.py
# Cloned from it. 
TASK_EMBEDDING_MAP = {
    'ContributionRepo': ['contributor', 'repository'],
    'PRReviewer':       ['repository', 'pr', 'contributor'],
    'RepoMaintainer':   ['repository', 'contributor'],
    'SimDeveloper':     ['contributor', 'contributor'],
}

class EmbeddingNet(nn.Module): 

    def __init__(self, embedding_input_sizes, *, embedding_dim): 
        '''
        embedding_input_sizes: {
            'contributor':  int, 
            'repostiory':   int, 
            'pr':           int, 
            'issue':        int,
        }
        We don't check if `embedding_input_sizes` meets the requirment above, 
        the invoker should confirm on themselves.
        '''
        super(EmbeddingNet, self).__init__()
        self.embedding_dim = embedding_dim

        # This so called 'embeddings_layer', is just what we want finally.
        self.embeddings_layer = nn.ModuleDict({
            'contributor':  nn.Linear(embedding_input_sizes['contributor'], embedding_dim),
            'repository':   nn.Linear(embedding_input_sizes['repository'],  embedding_dim),
            'pr':           nn.Linear(embedding_input_sizes['pr'],          embedding_dim),
            'issue':        nn.Linear(embedding_input_sizes['issue'],       embedding_dim),
        })
        # Now, let's build the remains. 
        self.remain_layers = nn.ModuleDict({
            task_name: nn.Sequential(
                nn.Linear(embedding_dim * len(entity_names), 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            for task_name, entity_names in TASK_EMBEDDING_MAP.items()
        })

    def forward(self, task_one_hot: torch.Tensor, embs: List[torch.Tensor]):  

        # 1. Convert embeddings into target embeding size, 
        # with embeddings_layer
        embs = [
            self.embeddings_layer[entity_name](emb) 
            for entity_name, emb in 
            zip(ALL_ENTITIES, embs)
        ]

        # 2. Concat those layers, input them to the hidden layer
        #    to get a hidden embedding with size 512
        '''
        Going though the whole experiment you can see that one-hot encoding is not necessary, 
        but what if we change another architecure of EmbeddingNet?
        E.g., Consider such achievement: 
            1. Use a large embedding layer as embedding layer; 
            2. Concat of all (entity) types of embeddings, input them to the layer, 
               to get a embedding with size `embedding_dim * len(ALL_ENTITIES)`;
            3. concat such emdedding with task_one_hot; 
            4. input the result to another layer, to get a hidden layers with size 512; 
            5. do the remains.
        (step 1~2 covers real step 1; step 3~4 covers real step 2)

        We reserve one-hot input for such usage. 
        (Sounds suspicious, why don't input task indices and do one-hot encoding here? ...anyway)
        '''
        _, indices = torch.max(task_one_hot, dim=1)


        data = []
        for task_idx, emb in zip(indices, zip(*embs)):
            # emb: [embedding_of_a(entity) for entity in ALL_ENTITIES]
            task_name = ALL_TASKS[task_idx]
            embs_in = [
                emb[ALL_ENTITIES.index(entity_name)]
                for entity_name in TASK_EMBEDDING_MAP[task_name]
            ] # embs_in: the wanted result for task_name
            embs_out = self.remain_layers[task_name](torch.cat(embs_in, dim=-1))
            data.append(embs_out)

        x = torch.stack(data)
        return x.squeeze()



class EmbeddingNetDataset(Dataset): 

    '''
    @see also: 
        * Comparisons/experiments/delta/train_nn_10fold.py: MyDataset
    '''
    def __init__(
                self, 
                embeddings,  
                task_sample_indices: dict,
        ) -> None:
        '''
        task_sample_indices: {
            'ContributionRepo': str
            'PRReviewer':       str
            'RepoMaintainer':   str
            'SimDeveloper':     str
        }.
        This param indicates the indice files for each task.
        '''

        super().__init__()
        self.embeddings = embeddings

        self.data = []  # HAHA, take this! 

        '''
        Now, let's load those f*cking indices and organize embeddings 
        <b>grouped by task names</b>. 
        As this is an dataset for EmbeddingNet, 
        we write down those task names explicitly. 

        > [WARNING] we don't check whether `task_sample_indices` 
        > and if the file contents are valid or not. 
        > (`task_sample_indices` don't have all four tasks, the 
        >  element size of file content don't match the code below, etc.)
        > Use this at your own risk.
        '''

        # region # load indices and embeddings 

        # kind of ugly approach(disobey the DRY principle), 
        # but codes are mostly generated by copilot. 
        # just take it and stay calm: don't obsess the code cleaness too much.

        # task 1/4. ContributionRepo
        with open(task_sample_indices['ContributionRepo'], 'r', encoding='utf-8') as f: 
            for sample in json.load(f):
                contributor_idx, pos_repo_idx, neg_repo_idx = sample
                contributor, pos, neg = [
                    self.embeddings['contributor'][contributor_idx],
                    self.embeddings['repository'][pos_repo_idx],
                    self.embeddings['repository'][neg_repo_idx],
                ]
                self.data.append(['ContributionRepo', True,  [contributor, pos]])
                self.data.append(['ContributionRepo', False, [contributor, neg]])
        # task 2/4. PRReviewer
        with open(task_sample_indices['PRReviewer'], 'r', encoding='utf-8') as f: 
            for sample in json.load(f):
                repo_idx, pr_idx, pos_contributor_idx, neg_contributor_idx = sample
                repo, pr, pos, neg = [
                    self.embeddings['repository'][repo_idx],
                    self.embeddings['pr'][pr_idx],
                    self.embeddings['contributor'][pos_contributor_idx],
                    self.embeddings['contributor'][neg_contributor_idx],
                ]
                self.data.append(['PRReviewer', True,  [repo, pr, pos]])
                self.data.append(['PRReviewer', False, [repo, pr, neg]])
        # task 3/4. RepoMaintainer
        with open(task_sample_indices['RepoMaintainer'], 'r', encoding='utf-8') as f: 
            for sample in json.load(f):
                repo_idx, pos_contributor_idx, neg_contributor_idx = sample
                repo, pos, neg = [
                    self.embeddings['repository'][repo_idx],
                    self.embeddings['contributor'][pos_contributor_idx],
                    self.embeddings['contributor'][neg_contributor_idx],
                ]
                self.data.append(['RepoMaintainer', True,  [repo, pos]])
                self.data.append(['RepoMaintainer', False, [repo, neg]])
        # task 4/4. SimDeveloper
        with open(task_sample_indices['SimDeveloper'], 'r', encoding='utf-8') as f: 
            for sample in json.load(f):
                contributor_idx, pos_contributor_idx, neg_contributor_idx = sample
                contributor, pos, neg = [
                    self.embeddings['contributor'][contributor_idx],
                    self.embeddings['contributor'][pos_contributor_idx],
                    self.embeddings['contributor'][neg_contributor_idx],
                ]
                self.data.append(['SimDeveloper', True,  [contributor, pos]])
                self.data.append(['SimDeveloper', False, [contributor, neg]])
        # endregion

        # shuffle
        random.shuffle(self.data) 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


    @staticmethod 
    def load_embeddings(filepath): 
        print('loading embeddings from graph...', file=sys.stderr)
        graph = load_graphs(filepath)[0][0]
        return {
            it: graph.nodes[it].data['feat']
            for it in ALL_ENTITIES
        }

    @staticmethod
    def embeddings_size(embeddings): 
        return {
            entity: embeddings[entity].shape[1]
            for entity in ALL_ENTITIES
        }


def create_collate_fn(embedding_input_sizes):
    '''
    embedding_input_sizes: {
        'contributor':  int, 
        'repostiory':   int, 
        'pr':           int, 
        'issue':        int,
    }
    '''

    def collate_fn(batch): 
        ret_task_onehot = []
        ret_label = []
        ret_embeddings = []

        for task_name, is_positive, embs in batch: 
            one_hot = F.one_hot(torch.LongTensor([ALL_TASKS.index(task_name)]), len(ALL_TASKS))
            ret_task_onehot.append(one_hot)

            ret_label.append(int(is_positive))

            curr_embeddings = {
                entity_name: emb
                for entity_name, emb in 
                zip(TASK_EMBEDDING_MAP[task_name], embs)
            }
            curr_embeddings = [
                curr_embeddings.get(
                    entity_name, 
                    torch.zeros(embedding_input_sizes[entity_name]))
                for entity_name in ALL_ENTITIES
            ]
            ret_embeddings.append(curr_embeddings)

        return \
                torch.cat(ret_task_onehot), \
                [
                    torch.cat([it.unsqueeze(0) for it in emb])
                    for emb in zip(*ret_embeddings)
                ], \
                torch.LongTensor(ret_label), 

    return collate_fn

def train(
        model: nn.Module, 
        dataloader, 
        *, 
        epochs: int,
        device: str,
        model_path: str,
): 
    # These params are cloned from: 
    # `Comparisons/experiments/delta/train_nn_10fold.py`

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCELoss()

    for _ in tqdm(range(epochs)):
        running_loss = 0.0
        model.train()

        for (task_one_hot, embs, labels) in dataloader:
            task_one_hot = task_one_hot.to(device)
            embs = [emb.to(device) for emb in embs]
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(task_one_hot, embs)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    torch.save(model.state_dict(), model_path)
        
def test(
        model: nn.Module, 
        dataloader, 
        *, 
        device: str,
        model_path: str,
): 
    '''
    Mostly copied from:
    `Comparisons/experiments/delta/train_nn_10fold.py`
    '''
    def metric(pos_right, neg_right, pos_total, neg_total):
        precision = (pos_right) / (pos_right + neg_total - neg_right)
        recall = (pos_right) / (pos_right + pos_total - pos_right)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    model.load_state_dict(torch.load(model_path))
    model.eval()
    pos_rights = 0
    neg_rights = 0
    pos_totals = 0
    neg_totals = 0
    with torch.no_grad():
        for (task_one_hot, embs, labels) in dataloader:
            task_one_hot = task_one_hot.to(device)
            embs = [emb.to(device) for emb in embs]
            labels = labels.to(device)

            logits = model(task_one_hot, embs)

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
    print(f'Test: {precision = }, {recall = }, {f1 = }')
    return precision, recall, f1
    

'''
|-------------|------|-------------|
| Entity Name | #emb | #emb w/o mp |
|-------------|------|-------------|
| contributor | 2304 | 2048        |
| repository  | 2304 | 2048        |
| pr          | 1792 | 1536        |
| issue       | 1024 | 768         | 
|-------------|------|-------------|
'''

def main(device, model_flp, embedding_flp, epochs): 

    embeddings = EmbeddingNetDataset.load_embeddings(embedding_flp)

    train_dataset = EmbeddingNetDataset(embeddings, {
        task_name: f'./data/{task_name}/train.json'
        for task_name in ALL_TASKS        
    })
    test_dataset = EmbeddingNetDataset(embeddings, {
        task_name: f'./data/{task_name}/test.json'
        for task_name in ALL_TASKS        
    })
    
    embeddings_size = train_dataset.embeddings_size(embeddings)

    
    net = EmbeddingNet(embeddings_size, embedding_dim=512)
    net = net.to(device)

    train_dataloader = DataLoader(
            train_dataset, 
            batch_size=32,
            shuffle=False, 
            collate_fn=create_collate_fn(embeddings_size))
    train(
        net, train_dataloader, 
        epochs=epochs, 
        model_path=model_flp,
        device=device
    )

    test_dataloader = DataLoader(
            test_dataset, 
            batch_size=32,
            shuffle=False, 
            collate_fn=create_collate_fn(embeddings_size))
    prec, recall, f1 = test(net, test_dataloader, model_path=model_flp, device=device)
    print(f'Test: {prec = }, {recall = }, {f1 = }')


CONFIG = {
    'v1': {
        'emb_v1': {
            'with_mp': {
                'device': 'cuda:0',
                'model_flp': 'bin/embedding_net/emb_v1/embedding_net.pth',
                'embedding_flp': '../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_with_metapath.bin',
                'epochs': 60,
            }, 
            'without_mp': {
                'device': 'cuda:1',
                'model_flp': 'bin/embedding_net/emb_v1/embedding_net_wo_mp.pth',
                'embedding_flp': '../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_without_metapath.bin',
                'epochs': 60,
            }
        },
        'emb_v2': {
            'with_mp': {
                'device': 'cuda:0',
                'model_flp': 'bin/embedding_net/emb_v2/embedding_net.pth',
                'embedding_flp': '../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_with_metapath_v2.bin',
                'epochs': 60,
            }, 
            'without_mp': {
                'device': 'cuda:1',
                'model_flp': 'bin/embedding_net/emb_v2/embedding_net_wo_mp.pth',
                'embedding_flp': '../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_without_metapath_v2.bin',
                'epochs': 60,
            }
        }
    }
}

if __name__ == '__main__': 
    main(**CONFIG['v1']['emb_v1']['without_mp'])
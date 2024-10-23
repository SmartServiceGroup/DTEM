
'''
Copied from "RecommendationTasks/RepoMaintainer/train_nn_10fold.py", modified.
'''

import json, sys
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from dgl import load_graphs


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
                torch.cat([embedding_a[entity_a_idx], embedding_b[pos_entity_b_idx]], dim=-1).numpy().tolist(),
                1,
            ])
            self.data.append([
                torch.cat([embedding_a[entity_a_idx], embedding_b[neg_entity_b_idx]], dim=-1).numpy().tolist(),
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
    precision = (pos_right) / (pos_right + neg_total - neg_right)
    recall = (pos_right) / (pos_right + pos_total - pos_right)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def load_embeddings(dst_graph_file: str, entity_names: list): 
    '''
        ensure the entity_names are availible. 
        Invoker's job. 
    '''

    # dst_graph_file = "./full_graph/structure_graph_with_average_feature_without_metapath.bin"
    # dst_graph_file = "./full_graph/structure_graph_with_average_feature_with_metapath.bin"
    graph = load_graphs(dst_graph_file)[0][0]

    return [
        graph.nodes[it].data['feat']
        for it in entity_names
    ]


if __name__ == "__main__":
    '''
        usage: 
            python3 train_nn_10fold.py [dst_file] [node_embedding_path] [entity_names] [task_name]

        e.g.
            python3 train_nn_10fold.py \
                result_RepoMaintainer_wo_mp.txt \
                ../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_without_metapath.bin \
                repository,contributor \
                RepoMaintainer

        HERE ARE SOME parameter constitutes: 

        |----------|---------------------|---------------------------|------------------|
        | dst_file | node_embedding_path | entity_names              | task_name        |
        |----------|---------------------|---------------------------|------------------|
        | [2]      | [1]                 | contributor,repository    | ContributionRepo |
        |          |                     | repository,contributor    | RepoMaintainer   |
        |          |                     | contributor,contributor   | SimDeveloper     |
        |----------|---------------------|---------------------------|------------------|

        [1] can either be ../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_with{,out}_metapath.bin 
        [2] suggested name: result_{task_name}_(wo_)?_mp.txt
    '''

    K = 10
    dst_file = sys.argv[1]
    node_embedding_path = sys.argv[2]
    entity_names = sys.argv[3].split(',')
    task_name = sys.argv[4]

    device = sys.argv[5] if len(sys.argv) > 5 else '1'
    device = torch.device(f'cuda:{device}')
    
    stat_precision = []
    stat_recall = []
    stat_f1 = []

    # Step 1. Create dataset
    embeddings = load_embeddings(node_embedding_path, entity_names)


    emb_a, emb_b = embeddings
    embedding_dim = emb_a.shape[1] + emb_b.shape[1]   # total input dim
    
    for i in range(K):
        train_sample_path = f'./data/{task_name}/10fold/train{i}.json'
        test_sample_path = f'./data/{task_name}/10fold/test{i}.json'
        model_path = f'./bin/{task_name}_{dst_file}_{i:02d}.bin'

        train_dataset = MyDataset(samples=train_sample_path, embedding_a=emb_a, embedding_b=emb_b)
        test_dataset  = MyDataset(samples=test_sample_path, embedding_a=emb_a, embedding_b=emb_b)
    
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_dataloader  = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

        model = Net(total_dim=embedding_dim)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.BCELoss()
        best_f1 = 0
        epochs = 60

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

        torch.save(model.state_dict(), model_path)
        # Test
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

    with open(dst_file, 'w') as f:
        f.write(str(stat_precision) + '\n')
        f.write(str(stat_recall) + '\n')
        f.write(str(stat_f1) + '\n')
        f.write("Average precision: " + str(avg_precision) + '\n')
        f.write("Average recall: " + str(avg_recall) + '\n')
        f.write("AVerage f1: " + str(avg_f1) + '\n')


'''
This table gives out params run on this script. 

|-----------------------------------|--------------------------------------------------------------------------------------------------|---------------------------|------------------|--------|
| output_file                       | embedding_filepath                                                                               | entity_names              | task_name        | device |
|-----------------------------------|--------------------------------------------------------------------------------------------------|---------------------------|------------------|--------|
| result_RepoMaintainer_wo_mp.txt   | ../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_without_metapath.bin | repository,contributor    | RepoMaintainer   |        |
| result_RepoMaintainer_mp.txt      | ../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_with_metapath.bin    | repository,contributor    | RepoMaintainer   |        |
| result_SimDeveloper_wo_mp.txt     | ../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_without_metapath.bin | contributor,contributor   | SimDeveloper     |        |
| result_SimDeveloper_wo_mp.txt     | ../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_without_metapath.bin | contributor,contributor   | SimDeveloper     |        |
| result_ContributionRepo_wo_mp.txt | ../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_without_metapath.bin | contributor,repository    | ContributionRepo | 0      |
| result_ContributionRepo_mp.txt    | ../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_with_metapath.bin    | contributor,repository    | ContributionRepo | 0      |
|-----------------------------------|--------------------------------------------------------------------------------------------------|---------------------------|------------------|--------|

for example, for line 1, we actually run: 

python3 train_nn_10fold.py \
    result_RepoMaintainer_wo_mp.txt \
    ../../../GNN/DataPreprocess/full_graph/structure_graph_with_average_feature_without_metapath.bin \
    repository,contributor \
    RepoMaintainer
'''
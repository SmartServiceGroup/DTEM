import json, sys
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# 10折交叉验证实验
# 在train_nn.py的基础上进行。原先按照8：1：1的比例划分了数据集，验证集在每一个epoch后都被评估一遍，测试集最后被评估一遍。
# 交叉验证中，按照9：1划分了数据集，产生10个训练集-测试集。
# 这个脚本分别训练10个模型，取测试集上的平均表现。
# python train_nn_10fold.py [输出文件] [图嵌入文件] [输出模型文件前缀] [嵌入向量长度]

class Net(nn.Module):
    def __init__(self, embedding_dim):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.embedding_dim * 3, 512)
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
                torch.cat([repo_node_embedding[repo_idx], pr_embedding[pr_idx], contributor_embedding[pos_reviewer_idx]], dim=-1).numpy().tolist(),
                1,
            ])
            self.data.append([
                torch.cat([repo_node_embedding[repo_idx], pr_embedding[pr_idx], contributor_embedding[neg_reviewer_idx]], dim=-1).numpy().tolist(),
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

if __name__ == "__main__":
    K = 10
    dst_file = sys.argv[1]
    embed_dim = int(sys.argv[4])
    device = torch.device('cuda:1')
    
    stat_precision = []
    stat_recall = []
    stat_f1 = []
    
    # Please replace the path with your own path
    node_embedding_path = sys.argv[2]
    all_embedding = torch.load(node_embedding_path)
    repo_node_embedding = all_embedding["repository"]
    contributor_node_embedding = all_embedding["contributor"]
    pr_node_embedding = all_embedding["pr"]
     
    
    for i in range(K):
        train_sample_path = './data/10fold/train{}.json'.format(i)
        test_sample_path = './data/10fold/test{}.json'.format(i)
        model_path = './bin/{}{}.bin'.format(sys.argv[3], i)
    
        train_dataset = MyDataset(samples=train_sample_path, repo_node_embedding=repo_node_embedding, pr_embedding=pr_node_embedding, contributor_embedding=contributor_node_embedding)
        test_dataset = MyDataset(samples=test_sample_path, repo_node_embedding=repo_node_embedding, pr_embedding=pr_node_embedding, contributor_embedding=contributor_node_embedding)
    
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

        model = Net(embedding_dim=embed_dim)
        model.to(device)
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
        model.load_state_dict(torch.load(model_path))
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
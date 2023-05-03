import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

class Net(nn.Module):
    def __init__(self, embedding_dim):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.embedding_dim * 2, 512)
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
    def __init__(self, samples, node_embedding_obj) -> None:
        super().__init__()
        self.data = []
        if isinstance(samples, str):
            with open(samples, "r", encoding="utf-8") as inf:
                samples = json.load(inf)
        for sample in samples:
            assert len(sample) == 3
            src_idx, pos_idx, neg_idx = sample
            self.data.append([
                torch.cat([node_embedding_obj[src_idx], node_embedding_obj[pos_idx]], dim=-1).numpy().tolist(),
                1,
            ])
            self.data.append([
                torch.cat([node_embedding_obj[src_idx], node_embedding_obj[neg_idx]], dim=-1).numpy().tolist(),
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
    train_sample_path = "./data/train.json"
    valid_sample_path = "./data/valid.json"
    test_sample_path = "./data/test.json"
    model_path = "./bin/model.bin"

    node_embedding_path = "../../GNN/HetSAGE/node_embedding/HetSAGE_node_embedding.bin"
    node_embedding = torch.load(node_embedding_path)["contributor"]
    
    train_dataset = MyDataset(samples=train_sample_path, node_embedding_obj=node_embedding)
    valid_dataset = MyDataset(samples=valid_sample_path, node_embedding_obj=node_embedding)
    test_dataset = MyDataset(samples=test_sample_path, node_embedding_obj=node_embedding)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = Net(embedding_dim=512)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCELoss()
    best_f1 = 0
    epochs = 60

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, (x, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}: loss={running_loss / len(train_dataloader)}")

        # Validation
        model.eval()
        pos_rights = 0
        neg_rights = 0
        pos_totals = 0
        neg_totals = 0

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(eval_dataloader):
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
        print(f"Epoch {epoch+1}: precision={precision}, recall={recall}, f1={f1}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_path)
            print("Model saved.")
    
    # Test  
    model.load_state_dict(torch.load(model_path))
    model.eval()
    pos_rights = 0
    neg_rights = 0
    pos_totals = 0
    neg_totals = 0
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_dataloader):
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
    print(f"Test: precision={precision}, recall={recall}, f1={f1}")



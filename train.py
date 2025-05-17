import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from data_loader import RNADataset, pad_collate
from model import RNABasePairPredictor
from structure_utils import evaluate_base_pair_prediction

def compute_pos_weight(dataset):
    pos, neg = 0, 0
    for _, label in dataset:
        pos += label.sum().item()
        neg += label.numel() - label.sum().item()
    return neg / (pos + 1e-8)

def train(num_epochs=20, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RNADataset("bpRNA_1m_90.fasta", "bpRNA_1m_90_DBNFILES", max_len=400, max_samples=500)
    val_size = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=pad_collate, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=pad_collate)

    model = RNABasePairPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    pos_weight = torch.tensor(compute_pos_weight(train_set), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y, lengths in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, lengths)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        print(f"\n📚 Epoch {epoch + 1} — Train loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        all_metrics = []
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, lengths)
                metrics = evaluate_base_pair_prediction(y[0].cpu(), pred[0].cpu(), threshold=0.7)
                all_metrics.append(metrics)
        avg = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        print(f"🧪 Validation — Precision: {avg['precision']:.4f}, Recall: {avg['recall']:.4f}, F1: {avg['f1']:.4f}, Accuracy: {avg['accuracy']:.4f}")

        if avg["f1"] > best_f1:
            best_f1 = avg["f1"]
            torch.save(model.state_dict(), "best_model.pt")

if __name__ == "__main__":
    train()

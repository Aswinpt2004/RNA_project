import torch
import torch.nn as nn

class RNABasePairPredictor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(2 * hidden_dim)
        self.linear_pairwise = nn.Linear(4 * hidden_dim, 128)
        self.scorer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, lengths):
        B, L, _ = x.shape
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(self.cnn(x).permute(0, 2, 1))
        x = self.dropout(lstm_out)
        x = self.norm(x)
        i = x.unsqueeze(2).expand(-1, L, L, -1)
        j = x.unsqueeze(1).expand(-1, L, L, -1)
        pair_repr = torch.cat([i, j], dim=-1)
        pair_repr = self.linear_pairwise(pair_repr)
        pair_scores = self.scorer(pair_repr).squeeze(-1)
        return pair_scores

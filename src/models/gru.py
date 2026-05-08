import torch
import torch.nn as nn

class DecoderGRU(nn.Module):
    def __init__(self, voca_size=10000, emd_size=256, hidden_size=512, max_len=20):
        super().__init__()
        self.h = nn.Linear(512, hidden_size)

        self.emd = nn.Embedding(voca_size, emd_size)
        self.gru = nn.GRU(emd_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, voca_size)

    def forward(self, features, captions):
        h = self.h(features).unsqueeze(0)  # (1, B, hidden)

        emd_cap = self.emd(captions)       # (B, T, emd)
        out, h = self.gru(emd_cap, h)      # (B, T, hidden)

        out = self.fc(out)                 # (B, T, vocab)

        return out
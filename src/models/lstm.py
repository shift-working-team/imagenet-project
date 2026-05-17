import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        voca_size=10000,
        emd_size=256,
        hidden_size=512,
        max_len=30
    ):
        super().__init__()

        self.max_len = max_len

        self.h = nn.Linear(512, hidden_size)
        self.c = nn.Linear(512, hidden_size)

        self.embedding = nn.Embedding(voca_size, emd_size)

        self.lstm = nn.LSTM(emd_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, voca_size)

    def forward(self, feature, caption):
        caption = caption.to(feature.device)

        h = self.h(feature).unsqueeze(0)
        c = self.c(feature).unsqueeze(0)

        input = self.embedding(caption)

        out, (h, c) = self.lstm(input, (h, c))

        out = self.fc(out)

        return out

    def generate(self, feature, start_token, end_token):
        device = feature.device
        start_token = start_token.to(device)
        
        h = self.h(feature).unsqueeze(0)
        c = self.c(feature).unsqueeze(0)

        generated = start_token.unsqueeze(1)

        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=feature.device)

        input = self.embedding(start_token).unsqueeze(1)

        for _ in range(self.max_len):
            out, (h, c) = self.lstm(input, (h, c))

            logits = self.fc(out).squeeze(1)

            pred = torch.argmax(logits, dim=1)

            pred[finished] = end_token

            generated = torch.cat([generated, pred.unsqueeze(1)], dim=1)

            finished |= (pred == end_token)

            if finished.all():
                break

            input = self.embedding(pred).unsqueeze(1)

        return generated[:, 1:].tolist()
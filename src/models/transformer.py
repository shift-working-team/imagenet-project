import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model, 2) * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, caption):
        return self.pe[:, :caption.size(1)] + caption
    

class DecoderTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, n_layers=4, voca_size=10000, max_len=30):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        self.img_proj = nn.Linear(512, d_model)
        self.embedding = nn.Embedding(voca_size, d_model)
        self.pos_end = PositionalEncoding(d_model, max_len)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.fc = nn.Linear(d_model, voca_size)

    def forward(self, feature, caption, pad_inx=0):
        memory = self.img_proj(feature).unsqueeze(1)
        input = self.embedding(caption)
        input = self.pos_end(input)

        T = caption.size(1)
        mask = torch.triu(torch.ones(T, T, device=input.device), diagonal=1).bool()
        pad_mask = (caption == pad_inx).to(input.device)

        out = self.transformer(tgt=input, memory=memory, tgt_mask=mask, tgt_key_padding_mask=pad_mask)

        preq = self.fc(out)

        return preq
    
    def generate(self, features, start_token, end_token):
        memory = self.img_proj(features).unsqueeze(1)
        generated = start_token.unsqueeze(1)
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=features.device)
        
        for _ in range(self.max_len):
            input = self.embedding(generated)
            input = self.pos_end(input)

            T = generated.size(1)
            mask = torch.triu(torch.ones(T, T, device=input.device), diagonal=1).bool()

            out = self.transformer(tgt=input, memory=memory, tgt_mask=mask)
            logits = self.fc(out)
            logits = logits[:, -1, :]

            pred = torch.argmax(logits, dim=1)
            pred[finished] = end_token

            generated = torch.concat([generated, pred.unsqueeze(1)],dim=1)

            finished |= (pred == end_token)

            if finished.all():
                break

        return generated[:, 1:].tolist()


        
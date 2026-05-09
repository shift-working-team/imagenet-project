import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        dev_term = torch.exp(torch.arange(0,d_model/2) * (math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * dev_term)
        pe[:, 1::2] = torch.cos(position * dev_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, caption):
        return self.pe[:, :caption.size(1)] + caption
    

class DecoderTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layer=4, voca_size=10000, max_len=30):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        self.img_proj = nn.Linear(512, d_model)
        self.embedding = nn.Embedding(voca_size, d_model)
        self.pos_end = PositionalEncoding(d_model, max_len)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layer)

        self.fc = nn.Linear(d_model, voca_size)

    def forward(self, feature, caption, pad_inx=0):
        memory = self.img_proj(feature).unsqueeze(1)
        input = self.embedding(caption)
        input = self.pos_end(input)

        T = caption.size(1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        pad_mask = (caption == pad_inx)

        out = self.transformer(tgt=input, memory=memory, tgt_mask=mask, tgt_key_padding_mask=pad_mask)

        preq = self.fc(out)

        return preq
    
    def generate(self, feature, start_token, end_token):
        memory = self.img_proj(feature).unsqueeze(1)
        generated = start_token.unsqueeze(1)

        for _ in range(self.max_len):
            input = self.embedding(generated)
            input = self.pos_end(input)

            out = self.transformer(tgt=input, memory=memory)
            logits = self.fc(out).squeeze(1)
            pred = torch.argmax(logits, dim=1)

            generated = torch.concat([generated, pred.unsqueeze(1)], dim=1)

            if pred.item() == end_token:
                break

        return generated[:, 1:]


        
import os
import torch
import torch.nn as nn
import math
from einops import rearrange
import matplotlib.pyplot as plt

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

class MHA(nn.Module):
    def __init__(self, d_model, nhead, drop_p):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(drop_p)

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.scale = math.sqrt(d_model // nhead)

    def forward(self, Q, K, V, mask=None):
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        Q = rearrange(Q, 'batch seq_len (nhead dim) -> batch nhead seq_len dim', nhead = self.nhead)
        K = rearrange(K, 'batch seq_len (nhead dim) -> batch nhead seq_len dim', nhead = self.nhead)
        V = rearrange(V, 'batch seq_len (nhead dim) -> batch nhead seq_len dim', nhead = self.nhead)

        attention_score = Q @ K.transpose(-1, -2) / self.scale # 
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask, -1e10)

        attention_weights = torch.softmax(attention_score, dim=-1) # B, nhead, seq_len, (seq_len or 49)
        attention_weights = self.dropout(attention_weights)

        attention = attention_weights @ V

        x = rearrange(attention, 'batch nhead seq_len dim -> batch seq_len (nhead dim)')
        x = self.fc_o(x)

        return x, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear = nn.Sequential(
                            nn.Linear(d_model, d_ff),
                            nn.ReLU(),
                            # nn.GELU(),
                            nn.Dropout(drop_p),
                            nn.Linear(d_ff, d_model)
                        )
        
    def forward(self, x):
        x = self.linear(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, drop_p):
        super().__init__()

        self.MHA = MHA(d_model, nhead, drop_p)
        self.MHA_LN = nn.LayerNorm(d_model)

        self.Cross_MHA = MHA(d_model, nhead, drop_p)
        self.Cross_MHA_LN = nn.LayerNorm(d_model)

        self.FFN = FeedForward(d_model, d_ff, drop_p)
        self.FFN_LN = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(drop_p)

    def forward(self, x, features, mask):
        residual, dec_weights = self.MHA(x, x, x, mask)
        residual = self.drop(residual)
        x = self.MHA_LN(x + residual)

        residual, enc_dec_weights = self.Cross_MHA(x, features, features, None)
        residual = self.drop(residual)
        x = self.Cross_MHA_LN(x + residual)

        residual = self.FFN(x)
        residual = self.drop(residual)
        x = self.FFN_LN(x + residual)

        return x, dec_weights, enc_dec_weights
    

class DecoderTransformer(nn.Module):
    def __init__(self, n_layers=4, nhead=8, d_model=512, d_ff=2048, voca_size=10000, max_len=30, drop_p=0.1):
        super().__init__()
        self.nhead = nhead
        self.max_len = max_len

        self.embedding = nn.Embedding(voca_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        # self.pos_enc = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff, drop_p) for _ in range(n_layers)])

        self.fc_out = nn.Linear(d_model, voca_size)

    def make_mask(self, T, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask
    
    def show_dec_atten(self, atten, generated_caption, n_layer, save_path): # layers, nhead, seq_len, seq_len)        
        atten = atten.mean(dim=1) # layers, seq_len, seq_len)
        atten = atten[n_layer-1] # seq_len, seq_len
        atten = atten.detach().cpu().numpy()

        seq_len = len(generated_caption)

        atten = atten[:seq_len, :seq_len]

        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(atten, cmap="bone")

        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))

        ax.set_xticklabels(generated_caption, rotation=45, ha="right")
        ax.set_yticklabels(generated_caption)

        plt.colorbar(im)

        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 저장
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()
    
    def show_cross_atten(self, atten, generated_caption, n_layer, image, save_path): # layers, nhead, seq_len, 49)
        import cv2
        import numpy as np

        # ------------------------
        # attention 전처리
        # ------------------------
        atten = atten.mean(dim=1) # layers, seq_len, seq_len)
        atten = atten[n_layer-1] # seq_len, seq_len

        atten = atten.detach().cpu().numpy()

        seq_len = len(generated_caption)
        atten = atten[:seq_len]

        # ------------------------
        # 이미지 준비
        # ------------------------
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()

            # (C,H,W) -> (H,W,C)
            image = image.permute(1, 2, 0).numpy()

            # normalize 복원 (ImageNet 기준)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            image = image * std + mean
            image = np.clip(image, 0, 1)

        H, W = image.shape[:2]

        # ------------------------
        # subplot 설정
        # ------------------------
        n_cols = 4
        n_rows = math.ceil(seq_len / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        axes = np.array(axes).reshape(-1)

        # ------------------------
        # 단어별 overlay
        # ------------------------
        for i in range(seq_len):

            # 49 -> 7x7
            num_patch = atten.shape[-1]
            side = int(math.sqrt(num_patch))

            heatmap = atten[i].reshape(side, side)

            # resize
            heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)

            # normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            ax = axes[i]

            # 원본 이미지
            ax.imshow(image)

            # heatmap overlay
            ax.imshow(heatmap, cmap="jet", alpha=0.45)

            ax.set_title(generated_caption[i])
            ax.axis("off")

        # 남는 subplot 제거
        for i in range(seq_len, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()
    

    def forward(self, features, x):
        mask = self.make_mask(x.shape[1], x.device)
        # pos = torch.arange(x.shape[1], device=x.device).expand_as(x) # expand_as(x) = x의 shape에 맞춰서 view해줌 (x.shape[1],) -> (B,x.shape[1])

        x = self.embedding(x)
        # x = x + self.pos_enc(pos)
        x = self.pos_enc(x)

        for layer in self.layers:
            x, dec_weights, enc_dec_weights = layer(x, features, mask)

        x = self.fc_out(x)

        return x
    
    def generate(self, features, start_token, end_token):
        generated = start_token.unsqueeze(1) # B, 1
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=features.device) # B,
        
        for _ in range(self.max_len - 1):
            # pos = torch.arange(generated.shape[1], device=generated.device).expand_as(generated) # expand_as(x) = x의 shape에 맞춰서 view해줌 (x.shape[1],) -> (B,x.shape[1])

            x = self.embedding(generated) # B, 1, d_model
            # x = x + self.pos_enc(pos)
            x = self.pos_enc(x) # B, 1, d_model

            mask = self.make_mask(generated.shape[1], generated.device)

            dec_atten = []
            enc_dec_atten = []
            # x->(B, 1, d_model), dec_weights->(B, nhead, seq_len, seq_len), enc_dec_weights->(B, nhead, seq_len, 49)
            for layer in self.layers:
                x, dec_weights, enc_dec_weights = layer(x, features, mask)

                dec_atten.append(dec_weights.detach().cpu()) #  layers*[B, nhead, seq_len, seq_len]
                enc_dec_atten.append(enc_dec_weights.detach().cpu()) # layers*[B, nhead, seq_len, 49]

            dec_atten = torch.stack(dec_atten, dim=1)
            enc_dec_atten = torch.stack(enc_dec_atten, dim=1)

            logits = self.fc_out(x) # B, 1, voca_size
            pred = torch.argmax(logits[:,-1,:], dim=-1) # B,

            pred[finished] = end_token

            generated = torch.cat([generated, pred.unsqueeze(1)], dim=1) # cat[(B, 1), (B, 1)] -> B, 2

            finished |= (pred == end_token)

            if finished.all():
                break

                # (B, seq_len-1), (B, layers, nhead, seq_len, seq_len), (B, layers, nhead, seq_len, 49)
        return generated[:,1:].tolist(), dec_atten, enc_dec_atten
    

    def generate_beam(self, features, start_token, end_token, beam_size):
        all_generated = []
        all_dec_atten = []
        all_enc_dec_atten = []
        for b in range(len(features)):
            feature = features[b].unsqueeze(0) # 1, seq, dim
            beams = [([start_token[b].item()], 0.0, None, None)] # seq, score

            for _ in range(self.max_len - 1):
                candidates = []
                for seq, score, prev_dec, prev_enc_dec in beams:
                    if seq[-1] == end_token:
                        candidates.append((seq, score, prev_dec, prev_enc_dec))
                        continue
                        
                    input_seq = torch.tensor(seq, device=feature.device).unsqueeze(0) # 1, seq

                    x = self.embedding(input_seq) # 1, seq, d_model
                    x = self.pos_enc(x) # 1, seq, d_model

                    mask = self.make_mask(input_seq.shape[1], input_seq.device)

                    dec_atten = []
                    enc_dec_atten = []
                    # x->(1, 1, d_model), dec_weights->(1, nhead, seq_len, seq_len), enc_dec_weights->(1, nhead, seq_len, 49)
                    for layer in self.layers:
                        x, dec_weights, enc_dec_weights = layer(x, feature, mask)

                        dec_atten.append(dec_weights.detach().cpu()) #  layers*[1, nhead, seq_len, seq_len]
                        enc_dec_atten.append(enc_dec_weights.detach().cpu()) #  layers*[1, nhead, seq_len, seq_len]

                    dec_atten = torch.stack(dec_atten, dim=1) # 1, layers, nhead, seq_len, seq_len
                    enc_dec_atten = torch.stack(enc_dec_atten, dim=1) # 1, layers, nhead, seq_len, 49

                    logits = self.fc_out(x) # 1, 1, voca_size

                    log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

                    topk_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

                    for k in range(beam_size):
                        token = topk_ids[0, k].item()
                        token_score = topk_probs[0, k].item()

                        candidates.append((seq + [token], score + token_score, dec_atten, enc_dec_atten))

                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

                if all(seq[-1] == end_token for seq, _, _, _ in beams):
                    break
            
            best_seq, _, best_dec_atten, best_enc_dec_atten = beams[0]

            all_generated.append(best_seq[1:]) # sos 제거
            all_dec_atten.append(best_dec_atten.squeeze(0))
            all_enc_dec_atten.append(best_enc_dec_atten.squeeze(0))

        return all_generated, all_dec_atten, all_enc_dec_atten
import torch
import torch.nn as nn

class DecoderLSTM(nn.Module): # MM
  def __init__(self, voca_size=10000, emd_size=256, hidden_size=512, max_len=30):
    super().__init__()
    self.max_len = max_len

    self.h = nn.Linear(512, hidden_size)
    self.c = nn.Linear(512, hidden_size)

    self.embedding = nn.Embedding(voca_size, emd_size)
    self.lstm = nn.LSTM(emd_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, voca_size)

  def forward(self, feature, caption):
    h = self.h(feature).unsqueeze(0) # 이미지 특성을 초기 h값으로 설정, 초기 문맥
    c = self.c(feature).unsqueeze(0) # 이미지 특성을 초기 c값으로 설정, 초기 문맥
    input = self.embedding(caption)

    out, (h, c) = self.lstm(input, (h, c))

    out = self.fc(out)

    return out
  
  def generate(self, feature, start_token, end_token):
    h = self.h(feature).unsqueeze(0) # 이미지 특성을 초기 h값으로 설정, 초기 문맥
    c = self.c(feature).unsqueeze(0) # 이미지 특성을 초기 c값으로 설정, 초기 문맥
    input = self.embedding(start_token).unsqueeze(1)

    generated_inx = []
    for _ in range(self.max_len):
      out, (h, c) = self.lstm(input, (h, c))
      logits = self.fc(out).squeeze(1)
      pred = torch.argmax(logits, dim=1)


      if pred.item() == end_token:
          break
    
      generated_inx.append(pred.item())

      input = self.embedding(pred).unsqueeze(1)
    
    return generated_inx

import torch.nn as nn

class DecoderLSTM(nn.Module): # MM
  def __init__(self, voca_size=10000, emd_size=256, hidden_size=512, max_len=20):
    super().__init__()
    self.h = nn.Linear(512, hidden_size)
    self.c = nn.Linear(512, hidden_size)

    self.emd = nn.Embedding(voca_size, emd_size)
    self.lstm = nn.LSTM(emd_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, voca_size)

  def forward(self, features, captions):
    h = self.h(features).unsqueeze(0) # 이미지 특성을 초기 h값으로 설정, 초기 문맥
    c = self.c(features).unsqueeze(0) # 이미지 특성을 초기 c값으로 설정, 초기 문맥

    emd_cap = self.emd(captions)
    out, (h, c) = self.lstm(emd_cap, (h, c))

    out = self.fc(out)

    return out
import sys
sys.path.append("/workspace/src/models")
import torch

# from 파일명 import 클래스명
from lstm import DecoderLSTM
from gru import DecoderGRU



feature = torch.randn([1,512])
caption = torch.tensor([[0, 1, 2, 3, 4]])

### LSTM Forward ###
model = DecoderLSTM()
out = model(feature, caption)
print(f'LSTM: {out.shape}')

### GRU Foreward ###
model = DecoderGRU()
out = model(feature, caption)
print(f'GRU: {out.shape}')
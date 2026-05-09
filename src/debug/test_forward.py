import sys
sys.path.append("/workspace/src/models")
import torch

# from 파일명 import 클래스명
from lstm import DecoderLSTM
from gru import DecoderGRU
from transformer import DecoderTransformer



feature = torch.randn([1,512])
caption = torch.tensor([[0, 1, 2, 3, 4]])


### LSTM Forward ###
model = DecoderLSTM()
out = model(feature, caption)
print(f'LSTM: {out.shape}')

### GRU Forward ###
model = DecoderGRU()
out = model(feature, caption)
print(f'GRU: {out.shape}')

### Transformer Forward ###
model = DecoderTransformer()
out = model(feature, caption, 0)
print(f'Transformer: {out.shape}')
import torch.nn as nn

from models.resnet18_seon import EncoderResnet18
from models.lstm import DecoderLSTM


class CNN_LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = EncoderResnet18()
        self.decoder = DecoderLSTM(
            voca_size=vocab_size
        )

    def forward(self, images, captions):
        _, embeddings = self.encoder(images)
        outputs = self.decoder(
            embeddings,
            captions
        )

        return outputs
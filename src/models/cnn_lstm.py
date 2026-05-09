import torch.nn as nn

from src.models.encoder import ResNet18Encoder
from src.models.lstm import DecoderLSTM


class CNN_LSTM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.decoder = DecoderLSTM(
            voca_size=vocab_size
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(
            features,
            captions
        )

        return outputs
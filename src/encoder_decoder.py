import torch
import torch.nn as nn


class EncodeDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(source)
        return self.decoder(target, encoder_outputs, hidden, teacher_forcing_ratio)
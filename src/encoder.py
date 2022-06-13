import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_len, embedding_dim, encoder_hidden_dim, n_layers=1, dropout_prob=0.5):
        super().__init__()

        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, n_layers, dropout=dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_batch):
        embedded = self.dropout(self.embedding(input_batch))
        outputs, hidden = self.rnn(embedded)

        return outputs, hidden

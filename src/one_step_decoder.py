import torch
import torch.nn as nn


class OneStepDecoder(nn.Module):
    def __init__(self, input_output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, attention, dropout_prob=0.5):
        super().__init__()

        self.output_dim = input_output_dim
        self.attention = attention

        self.embedding = nn.Embedding(input_output_dim, embedding_dim)

        # Add the encoder_hidden_dim and embedding_dim
        self.rnn = nn.GRU(encoder_hidden_dim + embedding_dim, decoder_hidden_dim)
        # Combine all the features for better prediction
        self.fc = nn.Linear(encoder_hidden_dim + decoder_hidden_dim + embedding_dim, input_output_dim)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, hidden, encoder_outputs):
        # Add the source len dimension
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        # Calculate the attention weights
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)

        # We need to perform the batch wise dot product.
        # Hence need to shift the batch dimension to the front.
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Use PyTorch's bmm function to calculate the
        # weight W.
        W = torch.bmm(a, encoder_outputs)

        # Revert the batch dimension.
        W = W.permute(1, 0, 2)

        # concatenate the previous output with W
        rnn_input = torch.cat((embedded, W), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        # Remove the sentence length dimension and pass them to the Linear layer
        predicted_token = self.fc(torch.cat((output.squeeze(0), W.squeeze(0), embedded.squeeze(0)), dim=1))

        return predicted_token, hidden, a.squeeze(1)

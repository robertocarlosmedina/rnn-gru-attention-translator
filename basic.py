import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F

from src.attention import Attention
from src.decoder import Decoder
from src.encoder import Encoder
from src.encoder_decoder import EncodeDecoder
from src.one_step_decoder import OneStepDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_datasets(batch_size=128):
    # Download the language files
    spacy_de = spacy.load('pt_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    # define the tokenizer
    def tokenize_de(text):
        return [token.text for token in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [token.text for token in spacy_en.tokenizer(text)]

    # Create the pytext's Field
    source = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
    target = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

    # Splits the data in Train, Test and Validation data
    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".cv", ".en"), fields=(source, target),
        test="test", path=".data/criolSet"
    )

    # Build the vocabulary for both the language
    source.build_vocab(train_data, min_freq=3)
    target.build_vocab(train_data, min_freq=3)

    # Create the Iterator using builtin Bucketing
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                          batch_size=batch_size,
                                                                          sort_within_batch=True,
                                                                          sort_key=lambda x: len(x.src),
                                                                          device=device)
    return train_iterator, valid_iterator, test_iterator, source, target


def create_model(source, target):
    # Define the required dimensions and hyper parameters
    embedding_dim = 256
    hidden_dim = 1024
    dropout = 0.5

    # Instantiate the models
    attention_model = Attention(hidden_dim, hidden_dim)
    encoder = Encoder(len(source.vocab), embedding_dim, hidden_dim)
    one_step_decoder = OneStepDecoder(len(target.vocab), embedding_dim, hidden_dim, hidden_dim, attention_model)
    decoder = Decoder(one_step_decoder, device)

    model = EncodeDecoder(encoder, decoder)

    model = model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters())

    # Makes sure the CrossEntropyLoss ignores the padding tokens.
    TARGET_PAD_IDX = target.vocab.stoi[target.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_IDX)

    return model, optimizer, criterion


def train(train_iterator, valid_iterator, source, target, epochs=10):
    model, optimizer, criterion = create_model(source, target)

    clip = 1

    for epoch in range(1, epochs + 1):
        pbar = tqdm(total=len(train_iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200)

        training_loss = []
        # set training mode
        model.train()

        # Loop through the training batch
        for i, batch in enumerate(train_iterator):
            # Get the source and target tokens
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()

            # Forward pass
            output = model(src, trg)

            # reshape the output
            output_dim = output.shape[-1]

            # Discard the first token as this will always be 0
            output = output[1:].view(-1, output_dim)

            # Discard the sos token from target
            trg = trg[1:].view(-1)

            # Calculate the loss
            loss = criterion(output, trg)

            # back propagation
            loss.backward()

            # Gradient Clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            training_loss.append(loss.item())

            pbar.set_postfix(
                epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}", refresh=True)
            pbar.update()

        with torch.no_grad():
            # Set the model to eval
            model.eval()

            validation_loss = []

            # Loop through the validation batch
            for i, batch in enumerate(valid_iterator):
                src = batch.src
                trg = batch.trg

                # Forward pass
                output = model(src, trg, 0)

                output_dim = output.shape[-1]

                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                # Calculate Loss
                loss = criterion(output, trg)

                validation_loss.append(loss.item())

        pbar.set_postfix(
            epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}, val loss= {round(sum(validation_loss) / len(validation_loss), 4)}",
            refresh=False)
        pbar.close()

    return model


if __name__ == '__main__':
    train_iterator, valid_iterator, test_iterator, source, target = get_datasets(batch_size=12)
    model = train(train_iterator, valid_iterator, source, target, epochs=1)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'source': source.vocab,
        'target': target.vocab
    }

    torch.save(checkpoint, 'checkpoints/nmt-model-gru-attention-25.pth')
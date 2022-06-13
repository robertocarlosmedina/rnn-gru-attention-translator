import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from termcolor import colored

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


EPOCHS = 1
CLIP = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Seq2Seq_Translator:

    # Download the language files
    spacy_de = spacy.load('pt_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def __init__(self) -> None:
        self.get_datasets()
        self.create_model()
        pass
        
    # define the tokenizer
    def tokenize_de(self, text):
        return [token.text for token in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [token.text for token in self.spacy_en.tokenizer(text)]

    def get_datasets(self, batch_size=128):
        
        # Create the pytext's Field
        self.source = Field(tokenize=self.tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
        self.target = Field(tokenize=self.tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

        # Splits the data in Train, Test and Validation data
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(
            exts=(".cv", ".en"), fields=(self.source, self.target),
            test="test", path=".data/criolSet"
        )

        # Build the vocabulary for both the language
        self.source.build_vocab(self.train_data, min_freq=3)
        self.target.build_vocab(self.train_data, min_freq=3)

        # Create the Iterator using builtin Bucketing
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device
        )

    def create_model(self):
        # Define the required dimensions and hyper parameters
        embedding_dim = 256
        hidden_dim = 1024
        dropout = 0.5

        # Instantiate the models
        attention_model = Attention(hidden_dim, hidden_dim)
        encoder = Encoder(len(self.source.vocab), embedding_dim, hidden_dim)
        one_step_decoder = OneStepDecoder(len(self.target.vocab), embedding_dim, hidden_dim, hidden_dim, attention_model)
        decoder = Decoder(one_step_decoder, device)

        self.model = EncodeDecoder(encoder, decoder)

        self.model = self.model.to(device)

        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters())

        # Makes sure the CrossEntropyLoss ignores the padding tokens.
        TARGET_PAD_IDX = self.target.vocab.stoi[self.target.pad_token]
        self.criterion = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_IDX)

        self.load_models()
    
    def load_models(self):
        print("=> Loading checkpoint")
        try:
            checkpoint = torch.load('checkpoints/nmt.model.gru-attention.pth.tar')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            print(colored("=> No checkpoint to Load", "red"))
    
    def save_model(self):
        print("=> Saving checkpoint")
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, 'checkpoints/nmt.model.gru-attention.pth.tar')

    def train_model(self):

        for epoch in range(1, EPOCHS + 1):
            progress_bar = tqdm(total=len(self.train_iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200)

            training_loss = []
            # set training mode
            self.model.train()

            # Loop through the training batch
            for i, batch in enumerate(self.train_iterator):
                # Get the source and target tokens
                src = batch.src
                trg = batch.trg

                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(src, trg)

                # reshape the output
                output_dim = output.shape[-1]

                # Discard the first token as this will always be 0
                output = output[1:].view(-1, output_dim)

                # Discard the sos token from target
                trg = trg[1:].view(-1)

                # Calculate the loss
                loss = self.criterion(output, trg)

                # back propagation
                loss.backward()

                # Gradient Clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)

                self.optimizer.step()

                training_loss.append(loss.item())

                progress_bar.set_postfix(
                    epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}", refresh=True)
                progress_bar.update()

            with torch.no_grad():
                # Set the model to eval
                self.model.eval()

                validation_loss = []

                # Loop through the validation batch
                for i, batch in enumerate(self.valid_iterator):
                    src = batch.src
                    trg = batch.trg

                    # Forward pass
                    output = self.model(src, trg, 0)

                    output_dim = output.shape[-1]

                    output = output[1:].view(-1, output_dim)
                    trg = trg[1:].view(-1)

                    # Calculate Loss
                    loss = self.criterion(output, trg)

                    validation_loss.append(loss.item())

            progress_bar.set_postfix(
                epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}, val loss= {round(sum(validation_loss) / len(validation_loss), 4)}",
                refresh=False)
            progress_bar.close()

        self.save_model()
    
    def display_attention(self, sentence, translation, attention):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        attention = attention.squeeze(1).cpu().detach().numpy()[:-1, 1:-1]

        cax = ax.matshow(attention, cmap='bone')

        ax.tick_params(labelsize=15)
        ax.set_xticklabels([''] + [t.lower() for t in sentence] + [''],
                           rotation=45)
        ax.set_yticklabels([''] + translation + [''])

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
        plt.close()
    
    def translate(self, display_attention=False):
        src = vars(self.test_data.examples[12])['src']
        trg = vars(self.test_data.examples[12])['trg']

        # Convert each source token to integer values using the vocabulary
        tokens = ['<sos>'] + [token.lower() for token in src] + ['<eos>']
        src_indexes = [self.source.vocab.stoi[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

        self.model.eval()

        # Run the forward pass of the encoder
        encoder_outputs, hidden = self.model.encoder(src_tensor)

        # Take the integer value of <sos> from the target vocabulary.
        trg_index = [self.target.vocab.stoi['<sos>']]
        next_token = torch.LongTensor(trg_index).to(device)

        attentions = torch.zeros(30, 1, len(src_indexes)).to(device)

        trg_indexes = [trg_index[0]]

        outputs = []
        with torch.no_grad():
            # Use the hidden and cell vector of the Encoder and in loop
            # run the forward pass of the OneStepDecoder until some specified
            # step (say 50) or when <eos> has been generated by the model.
            for i in range(30):
                output, hidden, a = self.model.decoder.one_step_decoder(next_token, hidden, encoder_outputs)

                attentions[i] = a

                # Take the most probable word
                next_token = output.argmax(1)

                trg_indexes.append(next_token.item())

                predicted = self.target.vocab.itos[output.argmax(1).item()]
                if predicted == '<eos>':
                    break
                else:
                    outputs.append(predicted)
                    
        print(colored(f'Ground Truth    = {" ".join(trg)}', 'green'))
        print(colored(f'Predicted Label = {" ".join(outputs)}', 'red'))

        predicted_words = [self.target.vocab.itos[i] for i in trg_indexes]

        if display_attention:
            self.display_attention(src, predicted_words[1:-1], attentions[:len(predicted_words) - 1])

        return predicted_words



if __name__ == '__main__':
    model = Seq2Seq_Translator()
    # model.train_model()
    # model.translate()
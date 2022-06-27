import math

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pyter import ter
from termcolor import colored

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
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
from src.grammar_checker import Grammar_checker
from src.utils import progress_bar

LEARNING_RATE = 3e-4
BATCH_SIZE = 10
EPOCHS = 250
CLIP = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Seq2Seq_Translator:

    # Download the language files
    spacy_models = {
        "en": spacy.load("en_core_web_sm"),
        "pt": spacy.load("pt_core_news_sm"),
        "cv": spacy.load("pt_core_news_sm"),
    }

    def __init__(self, source_languague: str, target_languague: str) -> None:
        self.source_languague = source_languague
        self.target_languague = target_languague
        self.get_datasets()
        self.create_model()
        self.writer = SummaryWriter()
        self.grammar = Grammar_checker()
        
    # define the tokenizer
    def tokenize_src(self, text):
        return [token.text for token in self.spacy_models[self.source_languague].tokenizer(text)]

    def tokenize_trg(self, text):
        return [token.text for token in self.spacy_models[self.target_languague].tokenizer(text)]

    def get_datasets(self):
        
        # Create the pytext's Field
        self.source = Field(tokenize=self.tokenize_src, init_token='<sos>', eos_token='<eos>', lower=True)
        self.target = Field(tokenize=self.tokenize_trg, init_token='<sos>', eos_token='<eos>', lower=True)

        # Splits the data in Train, Test and Validation data
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(
            exts=(f".{self.source_languague}", f".{self.target_languague}"), 
            fields=(self.source, self.target),
            test="test", path=".data/crioleSet"
        )

        # Build the vocabulary for both the language
        self.source.build_vocab(self.train_data, min_freq=3)
        self.target.build_vocab(self.train_data, min_freq=3)

        # Create the Iterator using builtin Bucketing
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=BATCH_SIZE,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device
        )

        print(colored("=> Data has been collected and processed", 'cyan'))

    def get_test_data(self) -> list:
        return [(test.src, test.trg) for test in self.test_data.examples[0:20]]

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Makes sure the CrossEntropyLoss ignores the padding tokens.
        TARGET_PAD_IDX = self.target.vocab.stoi[self.target.pad_token]
        self.criterion = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_IDX)

        self.load_models()
    
    def load_models(self):
        print(colored("=> Loading checkpoint", "cyan"))
        try:
            checkpoint = torch.load(
                f"checkpoints/gru-{self.source_languague}-{self.target_languague}.pth.tar",
                map_location='cpu')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print("\n\n", e, "\n\n")
            print(colored("=> No checkpoint to Load", "red"))
    
    def save_model(self):
        print(colored("=> Saving checkpoint", 'cyan'))
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(
            checkpoint, 
            f"checkpoints/gru-{self.source_languague}-{self.target_languague}.pth.tar")

    def show_train_metrics(self, epoch: int, train_loss: float, 
        train_accuracy: float, valid_loss: float, valid_accuracy:float) -> None:
        print(f' Epoch: {epoch:03}/{EPOCHS}')
        print(
            f' Train Loss: {train_loss:.3f} | Train Acc: {train_accuracy:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f' Val. Loss: {valid_loss:.3f} | Val Acc: {valid_accuracy:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}\n')
    
    def save_train_metrics(self, epoch: int, train_loss: float, 
            train_accuracy: float, valid_loss: float, valid_accuracy:float) -> None:
        """
            Save the training metrics to be ploted in the tensorboard.
        """
        # All stand alone metrics
        self.writer.add_scalar(
            f"Training Loss ({self.source_languague}-{self.target_languague})", 
            train_loss, global_step=epoch)
        self.writer.add_scalar(
            f"Training Accuracy ({self.source_languague}-{self.target_languague})", 
            train_accuracy, global_step=epoch)
        self.writer.add_scalar(
            f"Validation Loss ({self.source_languague}-{self.target_languague})", 
            valid_loss, global_step=epoch)
        self.writer.add_scalar(
            f"Validation Accuracy ({self.source_languague}-{self.target_languague})", 
            valid_accuracy, global_step=epoch)
        
        # Mixing Train Metrics
        self.writer.add_scalars(
            f"Training Loss & Accurary ({self.source_languague}-{self.target_languague})", 
            {"Train Loss": train_loss, "Train Accurary": train_accuracy},
            global_step=epoch
        )

        # Mixing Validation Metrics
        self.writer.add_scalars(
            f"Validation Loss & Accurary  ({self.source_languague}-{self.target_languague})", 
            {"Validation Loss": valid_loss, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )
        
        # Mixing Train and Validation Metrics
        self.writer.add_scalars(
            f"Train Loss & Validation Loss ({self.source_languague}-{self.target_languague})", 
            {"Train Loss": train_loss, "Validation Loss": valid_loss},
            global_step=epoch
        )
        self.writer.add_scalars(
            f"Train Accurary & Validation Accuracy ({self.source_languague}-{self.target_languague})",
            {"Train Accurary": train_accuracy, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )

    def train(self, epoch, progress_bar):

        target_count, correct_train = 0, 0
        training_loss = []
        training_accu = []
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

            # Calculate the Accuracy
            _, predicted = torch.max(output.data, 1)
            target_count += trg.size(0)
            correct_train += (trg == predicted).sum().item()
            training_accu.append((correct_train) / target_count)

            progress_bar.set_postfix(
                epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}, train accu: {sum(training_accu) / len(training_accu):.4f}", refresh=True)
            progress_bar.update()
        
        return sum(training_loss) / len(training_loss), sum(training_accu) / len(training_accu)
    
    def evaluate(self, epoch, progress_bar):

        target_count, correct_train, train_acc = 0, 0, 0

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
                # Calculate Accuracy
                _, predicted = torch.max(output.data, 1)
                target_count += trg.size(0)
                correct_train += (trg == predicted).sum().item()
                train_acc += (correct_train) / target_count

            progress_bar.set_postfix(
                epoch=f" {epoch}, val loss= {round(sum(validation_loss) / len(validation_loss), 4)}, val accu: {train_acc / i + 1:.4f}",
                refresh=False)
            progress_bar.close()
        
        return sum(validation_loss) / len(validation_loss), train_acc / len(self.valid_iterator)

    def train_model(self):

        for epoch in range(1, EPOCHS + 1):
            progress_bar = tqdm(
                total=len(self.train_iterator)+len(self.valid_iterator), 
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200
            )

            train_loss, train_accu = self.train(epoch, progress_bar)
            val_loss, val_accu = self.evaluate(epoch, progress_bar)

            self.show_train_metrics(epoch, train_loss, train_accu, val_loss, val_accu)
            self.save_train_metrics(epoch, train_loss, train_accu, val_loss, val_accu)
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
    
    def translate(self, sentence,  display_attention=False):

        # Convert each source token to integer values using the vocabulary
        if sentence is str:
            tokens = ['<sos>'] + [token.lower() for token in sentence.split()] + ['<eos>']
        else:
            tokens = ['<sos>'] + [token.lower() for token in sentence] + ['<eos>']

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

        predicted_words = [self.target.vocab.itos[i] for i in trg_indexes]

        if display_attention:
            print(tokens)
            self.display_attention(tokens[1:], predicted_words[1:-1], attentions[:len(predicted_words) - 1])

        return predicted_words[1:]

    def translate_sentence(self, sentence: str) -> str:
        predicted_words = self.translate(sentence)
        return self.untokenize_sentence(predicted_words)
    
    def generate_confusion_matrix(self) -> None:
        os.system("clear")
        print("\n                  CV Creole Translator Test ")
        print("-------------------------------------------------------------\n")
        sentence = str(input("  Sentence (cv): "))
        predicted_words = self.translate(sentence, True)
        print(
            colored(f'  Prediction (en): {self.untokenize_sentence(predicted_words)}', 'blue', attr=['bold'])
        )

    def untokenize_sentence(self, tokens: list) -> str:
        """
            Method to untokenuze the pedicted translation.
            Returning it on as an str.
        """
        if self.source_languague == "cv":
            tokens = self.remove_special_notation(tokens)
            translated_sentence = TreebankWordDetokenizer().detokenize(tokens)
            return self.grammar.check_sentence(translated_sentence)
        
        return " ".join(tokens)

    def test_model(self) -> None:
        test_data = self.get_test_data()
        os.system("clear")
        print("\n                  CV Creole Translator Test ")
        print("-------------------------------------------------------------\n")
        for data_tuple in test_data:
            src, trg = " ".join(
                data_tuple[0]), " ".join(data_tuple[1])
            translation = self.translate(src.split(" "))
            print(f'  Source (cv): {src}')
            print(colored(f'  Target (en): {trg}', attrs=['bold']))
            print(
                colored(f'  Predicted (en): {self.untokenize_sentence(translation)}\n', 'blue', attrs=['bold'])
            )

    def console_model_test(self) -> None:
        os.system("clear")
        print("\n                     CV Creole Translator ")
        print("-------------------------------------------------------------\n")
        while True:
            sentence = str(input(f'  Sentence (cv): '))
            translation = self.translate_sentence(sentence.split(" "))

            print(colored(f'  Predicted (en): {translation}\n', 'blue', attrs=['bold']))
    
    def count_hyperparameters(self) -> None:
        total_parameters =  sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(colored(f'\n==> The model has {total_parameters:,} trainable parameters\n', 'blue'))

    def remove_special_notation(self, sentence: list):
        return [token for token in sentence if token not in ["<unk>", "<eos>", "<sos>"]]

    def calculate_blue_score(self):
        """
            BLEU (bilingual evaluation understudy) is an algorithm for evaluating 
            the quality of text which has been machine-translated from one natural 
            language to another.
        """
        blue_scores = []
        len_test_data = len(self.test_data)

        for i, example in enumerate(self.test_data):
            src = vars(example)["src"]
            trg = vars(example)["trg"]
            predictions = []

            for _ in range(3):
                prediction = self.remove_special_notation(self.translate(src))
                predictions.append(prediction)

            # print(f'  Source (cv): {" ".join(src)}')
            # print(colored(f'  Target (en): {" ".join(trg)}', attrs=['bold']))
            # print(colored(f'  Predictions (en):', 'blue', attrs=['bold']))
            # [print(colored(f'      - {" ".join(prediction)}', 'blue', attrs=['bold'])) 
            #     for prediction in predictions]
            # print("\n")

            score = sentence_bleu(predictions, trg)
            blue_scores.append(score if score <= 1 else 1)

            progress_bar(i+1, len_test_data, f"BLUE score: {round(score, 8)}", "phases")

        score =  sum(blue_scores) /len(blue_scores)
        print(colored(f"\n\n==> Bleu score: {score * 100:.2f}\n", 'blue'))
    
    def calculate_meteor_score(self):
        """
            METEOR (Metric for Evaluation of Translation with Explicit ORdering) is 
            a metric for the evaluation of machine translation output. The metric is 
            based on the harmonic mean of unigram precision and recall, with recall 
            weighted higher than precision.
        """
        all_meteor_scores = []
        len_test_data = len(self.test_data)

        for i, example in enumerate(self.test_data):
            src = vars(example)["src"]
            trg = vars(example)["trg"]
            predictions = []

            for _ in range(4):
                prediction = self.remove_special_notation(self.translate(src))
                predictions.append(" ".join(prediction))

            score = meteor_score(predictions, " ".join(trg))
            all_meteor_scores.append(score)
            
            # print(f'  Source (cv): {" ".join(src)}')
            # print(colored(f'  Target (en): {" ".join(trg)}', attrs=['bold']))
            # print(colored(f'  Predictions (en):', 'blue', attrs=['bold']))
            # [print(colored(f'      - {prediction}', 'blue', attrs=['bold'])) for prediction in predictions]
            # print("\n")

            progress_bar(i+1, len_test_data, f"METEOR score: {round(score, 8)}", "phases")

        score = sum(all_meteor_scores)/len(all_meteor_scores)
        print(colored(f"\n\n==> Meteor score: {score * 100:.2f}\n", 'blue'))

    def calculate_ter(self):
        """
            TER. Translation Error Rate (TER) is a character-based automatic metric for 
            measuring the number of edit operations needed to transform the 
            machine-translated output into a human translated reference.
        """
        all_translation_ter = []
        len_test_data = len(self.test_data)

        for i, example in enumerate(self.test_data):
            src = vars(example)["src"]
            trg = vars(example)["trg"]

            prediction = self.remove_special_notation(self.translate(src))

            # print(f'  Source (cv): {" ".join(src)}')
            # print(colored(f'  Target (en): {" ".join(trg)}', attrs=['bold']))
            # print(colored(f'  Prediction (en): {" ".join(prediction)}\n', 'blue', attrs=['bold']))

            score = ter(prediction, trg)
            all_translation_ter.append(score)

            progress_bar(i+1, len_test_data, f"TER score: {round(score, 8)}", "phases")

        print(colored(f"\n\n==> TER score: {sum(all_translation_ter)/len(all_translation_ter) * 100:.2f}\n", 'blue'))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import unicodedata
import string
import re
import time
import math
from language import Lang


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def unicode_to_ascii(s):
    """
        Turn a Unicode string to plain ASCII, thanks to
        https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


# Lowercase, trim, and remove non-letter characters
def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2,):
    print("Reading lines...")

    # Read the train sentences
    train_src = open(f'.data/criolSet/train.{lang1}', encoding='utf-8').\
        read().strip().split('\n')
    train_trg = open(f'.data/criolSet/train.{lang2}', encoding='utf-8').\
        read().strip().split('\n')
    # Read to test sentences
    test_src = open(f'.data/criolSet/test.{lang1}', encoding='utf-8').\
        read().strip().split('\n')
    test_trg = open(f'.data/criolSet/test.{lang2}', encoding='utf-8').\
        read().strip().split('\n')
    # Read to val sentences
    val_src = open(f'.data/criolSet/val.{lang1}', encoding='utf-8').\
        read().strip().split('\n')
    val_trg = open(f'.data/criolSet/val.{lang2}', encoding='utf-8').\
        read().strip().split('\n')

    # Transforming the sentences in tuples according to their category
    train_data = [[normalize_string(src), normalize_string(trg)]
            for src, trg in zip(train_src, train_trg)]
    test_data = [[normalize_string(src), normalize_string(trg)]
            for src, trg in zip(test_src, test_trg)]
    val_data = [[normalize_string(src), normalize_string(trg)]
            for src, trg in zip(val_src, val_trg)]

    # pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, train_data, test_data, val_data


def filter_pair(p, max_lenght):
    return len(p[0].split(' ')) < max_lenght and \
        len(p[1].split(' ')) < max_lenght and \
        p[1].startswith(eng_prefixes)


def filter_pairs(pairs, max_length):
    return [pair for pair in pairs if filter_pair(pair, max_length)]


def prepare_data(lang1, lang2, max_length):
    input_lang, output_lang, train_data, test_data, val_dat = readLangs(lang1, lang2)
    print("Read %s sentence pairs" % len(train_data))
    train_data = filter_pairs(train_data, max_length)
    print("Trimmed to %s sentence pairs" % len(train_data))
    print("Counting words...")
    for data in train_data:
        input_lang.addSentence(data[0])
        output_lang.addSentence(data[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, train_data, test_data, val_dat


def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

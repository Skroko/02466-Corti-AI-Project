#%%
import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence
from collections import defaultdict
import nltk.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lexicon_path = "lexicon/librispeech-lexicon.txt"

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    #print("Raw Text Sequence: {}".format(text))
    #print("Phoneme Sequence: {}".format(phones))
    #sequence = np.array(
    #    text_to_sequence(
    #        phones, ["english_cleaners"]
    #    )
    #)

    #return np.array(sequence)
    return phones, text

#%%
preprocess_english("this is a test ? . ?")[0]
# %%


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle',encoding="utf-8")
fp = open("turing2.txt",encoding="utf-8")
data = fp.read()

# %%
sentences = tokenizer.tokenize(data)
sentences = [x.replace('\n',' ') for x in sentences]
sentences = [x.replace('\xa0',' ') for x in sentences]

#%%
short_test = sentences[:10]
#%%
# %%
df = defaultdict()
index = 0
for number, sentence in enumerate(short_test):
    current = sentences[number].split()
    if len(current) > 10: #checks if sentence is too long, then splits and
        while len(current) > 10:
            beginning = current[:10]
            current = current[10:]
            joined = ' '.join(beginning)
            df[index] = "|LJSpeech|"+preprocess_english(joined)[0] +"|"+preprocess_english(joined)[1]
            index += 1
        if len(current) > 0:
            end = ' '.join(current)
            df[index] = "|LJSpeech|"+preprocess_english(end)[0] +"|"+preprocess_english(end)[1]
            index += 1

    else:
        df[index] = "|LJSpeech|"+preprocess_english(sentence)[0] +"|"+preprocess_english(sentence)[1]
        index += 1
    print(df[number])

# %%
#Notice that it just continues writing if file already exists. its a feature, not a bug, definitely
with open('turing_phoneme.txt', 'a') as f:
    for key, value in df.items():
        f.write(str(key)+value)
        f.write('\n')
# %%

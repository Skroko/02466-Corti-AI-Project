#%%

import pandas as pd
import numpy as np
import jiwer
# %%
###### Creating transcript file of correct format needed for fine tuning #######
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.Strip(),
    #jiwer.RemoveWhiteSpace(replace_by_space=True),
    #jiwer.RemoveMultipleSpaces(),
    #jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])



with open('target.txt') as f:
    lines = f.readlines()

out_text =[]
for line in lines:
    out_text.append(transformation(line))

# %%

filenames = []

for count, ele in enumerate(out_text):
    filenames.append(f"{count}.wav {ele}")


# %%
with open("fastspeech.trans.txt", "w") as f:
    for filename in filenames:
        f.write(filename)
        f.write("\n")

# %%
####### create lexicon #########
#https://github.com/facebookresearch/fairseq/issues/2493
import os, codecs, re, pandas as pd
a = 'train.wrd'
b = 'valid.wrd'

df1 = pd.read_csv(a, header=None)
df2 = pd.read_csv(b, header=None)

df1.columns = ['raw']
df2.columns = ['raw']

df1 = df1.drop_duplicates('raw',keep='last')
df2 = df2.drop_duplicates('raw',keep='last')

sentence1 = df1['raw'].to_list()
sentence2 = df2['raw'].to_list()
sentence = sentence1 + sentence2

word = []
for x in sentence:
    tmp = x.split(' ')
    for y in tmp:
        if y not in word:
            word.append(y)

lexicon = []
for x in range(len(word)):
    wrd = word[x]
    temp = []
    for y in wrd:
        temp.append(y)
    result = ' '.join(temp) + ' |'
    lexicon.append(wrd + '\t ' + result)

file_to_save = 'lexicon.txt'
f=codecs.open(file_to_save,'a+','utf8')
for x in lexicon:
    f.write(x+'\n')
f.close()
# %%

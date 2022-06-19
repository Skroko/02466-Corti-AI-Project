#%%
#https://huggingface.co/Alvenir/wav2vec2-base-da-ft-nst
import soundfile as sf
import torch
import os
import pandas as pd
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Tokenizer, Wav2Vec2Processor, \
    Wav2Vec2ForCTC
import jiwer
from jiwer import wer

#%%
###### Loading model #####
def get_tokenizer(model_path: str) -> Wav2Vec2CTCTokenizer:
    return Wav2Vec2Tokenizer.from_pretrained(model_path)


def get_processor(model_path: str) -> Wav2Vec2Processor:
    return Wav2Vec2Processor.from_pretrained(model_path)


def load_model(model_path: str) -> Wav2Vec2ForCTC:
    return Wav2Vec2ForCTC.from_pretrained(model_path)


model_id = "Alvenir/wav2vec2-base-da-ft-nst"

model = load_model(model_id)
model.eval()
tokenizer = get_tokenizer(model_id)
processor = get_processor(model_id)


#%%
###### Loading labels and selecting audio files #######
os.chdir('C:\\Users\\rune7\\Documents\\GitHub\\wav2vec\\data\\cv-corpus-9.0-2022-04-27\\da') #path to validated lables
labels = pd.read_csv('validated.tsv', sep='\t')

validated_files = labels['path'].tolist()
validated_files = [s.replace(".mp3",".wav") for s in validated_files]

wav_dir = 'C:\\Users\\rune7\\Documents\\GitHub\\wav2vec\\data\\commonvoice'  # Path where the wavs are located
os.chdir(wav_dir)


#%%
######### Transcribing speech to text ##########
transcriptions = []
for audio_file in validated_files:
    audio, _ = sf.read(audio_file)

    input_values = processor(audio, return_tensors="pt", padding="longest", sampling_rate=16_000).input_values
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print(transcription)

    transcriptions.append(transcription[0])
# %%
#### Saving transcriptions as txt ###
os.chdir('C:\\Users\\rune7\\Documents\\GitHub\\wav2vec')
with open('CVtranscriptions.txt', 'w',encoding="utf-8") as f:
    for line in transcriptions:
       f.write(line)
       f.write('\n')

#%%
#### Loading transcriptions ###
os.chdir('C:\\Users\\rune7\\Documents\\GitHub\\wav2vec')
with open('CVtranscriptions.txt', encoding="utf-8") as f:
     transcriptions = f.readlines()


#%%
####### testing word error rate ###############
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    #jiwer.ExpandCommonEnglishContractions(),
    jiwer.Strip(),
    #jiwer.RemoveWhiteSpace(replace_by_space=True),
    #jiwer.RemoveMultipleSpaces(),
    #jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])

# %%
#### Transforming target and original text ####

#define ground truth

ground_truth = labels['sentence'].tolist()

hypothesis = transformation(transcriptions)
ground_truth = transformation(ground_truth)


#%%
def word_error(ground_truth, hypothesis):
    return(wer(ground_truth, hypothesis, ))


print(word_error(ground_truth, hypothesis))
#0.26255765199161424
# %%
######## Dividing up based on length #####
indexes = []

t=0
while t <= 100:
    if t == 100:
        temp_index = [i for i, x in enumerate(ground_truth) if (len(x) <= t+10000 and len(x) > t)]
        indexes.append([temp_index,t])
    else:
        temp_index = [i for i, x in enumerate(ground_truth) if (len(x) <= t+5 and len(x) > t)]
        indexes.append([temp_index,t])
    print(t)
    t+=5

truths = []
hypotheses = []
for index in indexes:
    truths.append([ground_truth[i] for i in index[0]])
    hypotheses.append([hypothesis[i] for i in index[0]])
#%%
#### Getting histogram of distribution ####
count = []
for i in range(len(indexes)):
    count.append(len(indexes[i][0]))

#%%
#### Calculating word error for each segment ####
errors = []

for i in range(len(truths)):
    errors.append(word_error(truths[i], hypotheses[i]))

sentence_max_length = [indexes[i][1] for i in range(len(indexes))]


# %%
### Creating dataframe and saving to csv ##
data = {'Sentence length': sentence_max_length,
        'Word Error Rate': errors,
        'Distribution': count}
df = pd.DataFrame(data)
print(df)

df.to_csv('WER.csv')
# %%

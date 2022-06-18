# https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html
#%%
from calendar import c
import os
import IPython
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchaudio
import jiwer
from jiwer import wer
from create_torchaudio_dataset import STTDataset
import nltk.data

#%%
#vctk = torchaudio.datasets.CMUARCTIC(root = "data", download = True) 

TEXTFILE = 'c:\\Users\\rune7\\Documents\\GitHub\\wav2vec\\target.txt'
AUDIO_DIRECTORY = 'c:\\Users\\rune7\\Documents\\GitHub\\wav2vec\\data\\ARCTIC\\renamed'
#^above is original audio. below is fastspech2 generated:
#AUDIO_DIRECTORY = 'c:\\Users\\rune7\\Documents\\GitHub\\wav2vec\\data\\Fastspeech2'

arctic = STTDataset(TEXTFILE,AUDIO_DIRECTORY)


#%%
arctic.__getitem__(1)

#%%
matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

#%%
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

#%%
#Transcribing audio files
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
            emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
            str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


out_text = []
for i in range(len(arctic)):
  waveform, transcript_true = arctic.__getitem__(i)
  waveform = waveform.to(device)


  with torch.inference_mode():
      emission, _ = model(waveform)



  decoder = GreedyCTCDecoder(labels=bundle.get_labels())
  transcript = decoder(emission[0])
  out_text.append(transcript)

#%%
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



# jiwer.wer(
#     ground_truth,
#     hypothesis,
#     truth_transform=transformation,
#     hypothesis_transform=transformation
# )

#%%
hypothesis = [text.replace("|"," ") for text in out_text]
ground_truth = arctic.text.values.tolist()

#%%
hypothesis = transformation(hypothesis)
ground_truth = transformation(ground_truth)


#%%
def word_error(ground_truth, hypothesis):
    return(wer(ground_truth, hypothesis, ))


print(word_error(ground_truth, hypothesis))

#%%

# cummulative_error = 0
# for i in range(len(ground_truth)):
#     cummulative_error += word_error(ground_truth[i],hypothesis[i])
# print(cummulative_error/len(ground_truth))




#ground_truth = [text.rstrip(",;.\-\?\!\s+())") for text in ground_truth]
#ground_truth.upper()

#print(transcript)
#IPython.display.Audio(SPEECH_FILE)
# with open('hypothesis.txt', 'w',encoding="utf-8") as f:
#     for line in out_text:
#       f.write(line)
#       f.write('\n')
  


# # %%
# #Getting text

# hypothesis = out_text

# #%%
# for i in range(len(lines)): #removing uwanted charachters
#   lines[i] = lines[i][16:-4]
# #%%
# with open('VCTK.txt', 'w',encoding="utf-8") as f:
#     for line in lines:
#       f.write(line)
#       f.write('\n')
  
# # %%
# results = []
# with open('VCTK.txt', encoding="utf-8") as f:
#     ground_truth = f.readlines()

# with open('fspeech2_guess.txt', encoding="utf-8") as f:
#     hypothesis = f.readlines()

# #%%


# # %%

# %%

# https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html
#%%
import os
import IPython
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchaudio
import jiwer
from jiwer import wer



def word_error(ground_truth, hypothesis):
    return(wer(ground_truth, hypothesis))



#%%
vctk = torchaudio.datasets.CMUARCTIC(root = "data", download = True) 

#%%
vctk.__getitem__(1)

#%%
matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

#%%
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

#%%
#Transcribing audio files
out_text = []
for i in range(len(vctk._walker)):
  waveform, sample_rate, transcript_true, utterance_id_ = vctk.__getitem__(i)
  waveform = waveform.to(device)

  if sample_rate != bundle.sample_rate:
      waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

  with torch.inference_mode():
      emission, _ = model(waveform)

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


  decoder = GreedyCTCDecoder(labels=bundle.get_labels())
  transcript = decoder(emission[0])
  out_text.append(transcript)

print(transcript)
#IPython.display.Audio(SPEECH_FILE)

# %%
#Getting text
with open(vctk._text) as f:
    lines = f.readlines()

for i in range(len(lines)): #removing uwanted charachters
  lines[i] = lines[i][16:-4]
#%%
with open('VCTK.txt', 'w',encoding="utf-8") as f:
    for line in lines:
      f.write(line)
      f.write('\n')
  
# %%
results = []
with open('VCTK.txt', encoding="utf-8") as f:
    ground_truth = f.readlines()

with open('fspeech2_guess.txt', encoding="utf-8") as f:
    hypothesis = f.readlines()


transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])

jiwer.wer(
    ground_truth,
    hypothesis,
    truth_transform=transformation,
    hypothesis_transform=transformation
)

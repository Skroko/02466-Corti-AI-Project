# https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html
#%%
import os
import IPython
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchaudio

#%%
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

#%%
matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

#%%
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
decoder = GreedyCTCDecoder(labels=bundle.get_labels())

#%%
# SPEECH_FILE = "./audio_data/2022-06-09 09-38-02.wav"
# waveform, sample_rate = torchaudio.load(SPEECH_FILE)

cmu_arctic = torchaudio.datasets.CMUARCTIC(root = "./audio_data", download = False)
waveform, sample_rate, transcript_true, utterance_id = cmu_arctic.__getitem__(1131)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

with torch.inference_mode():
    emission, _ = model(waveform)


decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])

print(transcript)
print(transcript_true.upper())
# IPython.display.Audio(SPEECH_FILE)

#%%
# test error rate
from jiwer import wer
import re
re_str = r"[.,']"

total_error_rate = [] 
wrong_words = []

for i in range(1131):
  waveform, sample_rate, transcript_true, utterance_id = cmu_arctic.__getitem__(i)
  waveform = waveform.to(device)
  if sample_rate != bundle.sample_rate:
      waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

  with torch.inference_mode():
      emission, _ = model(waveform)

  transcript = decoder(emission[0])
  xx = re.sub(re_str, "", re.sub(r"[|]", " ",transcript))
  yy = re.sub(re_str, "",transcript_true).upper()

  c = wer(xx,yy)
  total_error_rate += [c]
  if c > 0:
    print(f"{xx}\n{yy}")
    print("error",c , "time:",i)

print(sum(total_error_rate)/len(total_error_rate))


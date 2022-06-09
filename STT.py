# https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html
#%%
import os
import IPython
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio

#%%
matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

#%%
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

#%%
SPEECH_FILE = "./audio_data/The_fish_slapped_your_whale_mom_to_the_ground_The_fish_slapped_your_whale_mom_to_the_ground_The_fis.wav"
waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

#%%
# with torch.inference_mode():
#     features, _ = model.extract_features(waveform)

with torch.inference_mode():
    emission, _ = model(waveform)

#%%

## PLOT

# fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
# for i, feats in enumerate(features):
#     ax[i].imshow(feats[0].cpu())
#     ax[i].set_title(f"Feature from transformer layer {i+1}")
#     ax[i].set_xlabel("Feature dimension")
#     ax[i].set_ylabel("Frame (time-axis)")
# plt.tight_layout()
# plt.show()
#%%


#%%

## PLOT

# plt.imshow(emission[0].cpu().T)
# plt.title("Classification result")
# plt.xlabel("Frame (time-axis)")
# plt.ylabel("Class")
# plt.show()
# print("Class labels:", bundle.get_labels())

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
decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
#%%
print(transcript)
IPython.display.Audio(SPEECH_FILE)

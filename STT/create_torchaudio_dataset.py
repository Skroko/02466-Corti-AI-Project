#%%
from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchaudio



# %%
class STTDataset(Dataset):
    #with inspiration from: https://github.com/musikalkemist/pytorchforaudio/tree/main/04%20Creating%20a%20custom%20dataset

    def __init__(self, txt_file, audio_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the csv file with annotations.
            audio_dir (string): Directory with all the wav files.

        Takes folder of wav-files labelled 0,1,2...N corresponding to
        a TTS generated attempt and a txtfile with the targeted (source) text. 
        Outputs a torchaudio dataset w
        """
        self.text = df = pd.read_csv(txt_file, sep="/n", header = None)[0]
        self.audio_dir = audio_dir
        self.transform = transform

   

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        audio_sample_path = self._get_audio_sample_path(idx)
        text = self.text[idx]
        signal, sr = torchaudio.load(audio_sample_path)

        return signal, text

    def _get_audio_sample_path(self, idx):
        path = self.audio_dir + "\\" + str(idx) + ".wav"
        return path

#%%
if __name__ == "__main__":
    TEXTFILE = 'c:\\Users\\rune7\\Documents\\GitHub\\wav2vec\\target.txt'
    AUDIO_DIRECTORY = 'c:\\Users\\rune7\\Documents\\GitHub\\wav2vec\\data\\Fastspeech2'
    wav_dataset = STTDataset(TEXTFILE,AUDIO_DIRECTORY)


# %%

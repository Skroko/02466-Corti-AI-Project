#%%
##### batch converting mp3 to wav in order to use commonvoice dataset ######
import os
import glob
from pydub import AudioSegment
from pathlib import Path

mp3_dir = 'C:\\Users\\rune7\\Documents\\GitHub\\wav2vec\\data\\cv-corpus-9.0-2022-04-27\\da\\clips'  # Path where the mp3s are located

os.chdir(mp3_dir)
#%

#%%
files = os.listdir()
for file in files:
    wav_filename = os.path.splitext(os.path.basename(file))[0] + '.wav'
    my_file = Path(file)
    sound = AudioSegment.from_file(my_file)
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)
    sound.export(wav_filename, format='wav')

#%%

#%%
import pandas as pd
import os
import shutil

# %%
os.chdir('c:\\Users\\rune7\\Documents\\GitHub\\FastSpeech2\\media\\LJSpeech-1.1')
df = pd.DataFrame([line.strip().split(',') for line in open('metadata.csv', 'r',encoding='utf-8')])

df = df.iloc[::10, :] #keeping every 10th

df.to_csv('reduced_size_metadata.csv',index=False,header=False)

# %%
firstcol = df.iloc[:,0].values.tolist()
filenames =[]
for string in firstcol:
    filenames.append(string[:10])

# %%
#copying corpus data
src_path = 'c:\\Users\\rune7\\Documents\\GitHub\\FastSpeech2\\media\\LJSpeech-1.1\\wavs'
dst_path = 'c:\\Users\\rune7\\Documents\\GitHub\\FastSpeech2\\media\\LJSpeech-1.1\\reduced_wavs'

for name in filenames:
    shutil.copyfile(src_path + '\\' + name + '.wav',dst_path + '\\' + name + '.wav')

#%%
#copying raw data
src_path = 'c:\\Users\\rune7\\Documents\\GitHub\\FastSpeech2\\raw_data\\LJSpeech'
dst_path = 'c:\\Users\\rune7\\Documents\\GitHub\\FastSpeech2\\raw_data\\LJSpeech_reduced'

for name in filenames:
    shutil.copyfile(src_path + '\\' + name + '.wav',dst_path + '\\' + name + '.wav')
    shutil.copyfile(src_path + '\\' + name + '.lab',dst_path + '\\' + name + '.lab')


# %%

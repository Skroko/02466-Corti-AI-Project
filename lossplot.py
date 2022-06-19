# %%

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.dates as mdates
import seaborn as sns

def setup_plot():
    mpl.rcParams['lines.linewidth'] = 1
    #mpl.rcParams['font.family'] = 'Microsoft Sans Serif'
    mpl.rcParams['font.family'] = 'Arial'

    
    #these don't work for some reason
    #mpl.rcParams['axes.titleweight'] = 'bold'
    #mpl.rcParams['axes.titlesize'] = '90'
    
    sns.set_theme(style="white", palette='pastel', font = 'Arial', font_scale=3)

    #sns.set_theme(style="white", palette='pastel', font = 'Microsoft Sans Serif', font_scale=1)
    #myFmt = mdates.DateFormatter('%b #Y')
    
    print("Plot settings applied")

setup_plot()
# %%

path = "logged_data/16 batch"

# For code path
import os
import sys
from pathlib import Path

# This is used to read files in the module properly when the Main.py script is run from an external location.
#code_path = Path(*Path(os.path.realpath(sys.argv[0])).parts[:-1])
code_path = Path(os.getcwd())

# %%
path = code_path.joinpath(path)
#dirs = [os.path.join(path,d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))] 
# %%

files.sort(key = lambda x: int(x[:x.find('v')]))
# %%

means = np.zeros((len(files),5))

for i, file in enumerate(files):
    file_path = path.joinpath(file)
    df = pd.read_csv(file_path)
    means[i] = df.iloc[:].mean(0).to_numpy()

names = list(df.columns)

# %%

checkpoints = [int(x[:x.find('v')]) for x in files]

# brainlag here will fix tomorrow
for i in range(len(means)-1):
    plt.plot(checkpoints, means[:, i])

plt.legend(names)
plt.show()
# %%

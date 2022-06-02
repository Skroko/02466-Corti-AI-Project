# basic imports
import torch
import yaml


# our imports
from fastspeech2 import FastSpeech2
from loss import loss_class


# variabels
load_model = False # Change to True to load model
load_model_path = "" # TODO inseart path for saved models

# config path and loads
config_path = "./../config/model.yaml"
with open(config_path) as f:
    d = yaml.load(f,Loader=yaml.FullLoader)

print(d["transoformer"])

# get model
if load_model == True:
    None

else:
    model = FastSpeech2(config)

# train
Data = None
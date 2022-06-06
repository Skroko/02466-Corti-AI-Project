#!/bin/sh
#BSUB -J train-Fastspeech
#BSUB -o train-Fastspeech_%J.out
#BSUB -e train-Fastspeech_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -u rune7863@gmail.com
#BSUB -N
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 23:50
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.5.3-python-3.8.4

# load CUDA (for GPU support)
module load cuda/11.3

module load g2p-en/2.1.0

module load inflect/4.1.0

module load librosa/0.7.2

module load matplotlib/3.2.2

module load numba/0.48.0

module load numpy/1.19.2

module load pypinyin/0.39.0

module load pyworld/0.2.10

module load PyYAML/5.4.1

module load SoundFile/0.10.3.post1

module load tensorboard/2.2.2

module load tgt/1.4.4

module load torch/1.7.0

module load tqdm/4.46.1

module load Unidecode/1.1.1

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source fs-env/bin/activate

python train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml



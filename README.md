# CycleGAN-CV2-model
We did not reserve our pre-trained model in github.(Because our model is not sufficiently trained, if you want to get 
the best performance, you can change the epoch size in train.py. you should put your data in /data folder first. What
we put in/data folder is some test data.)

STEP 1:

Install these python libraries:

torch, librosa, tqdm, pyworld


STEP 2:

preprocess

command: python preprocess_training.py

STEP 3:

train model

command: python train.py

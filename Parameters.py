from CustomNeuralNetResNet import *
import torch
import random
import numpy as np

# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize the neural model model
outputs_number = 2
model = CustomNeuralNetResNet18(outputs_number)
#model = CustomNeuralNetResNet18(outputs_number)
model = model.to(device)

# number of images the neural net is training at the same time on
batch_size = 7

# nececary for backward propagation to calculate gradient descent
loss = torch.nn.CrossEntropyLoss()
# in fact, type of gradient descent's modification
optimizer = torch.optim.Adam(model.parameters(), lr=.1e-3)
# how many epochs should neural network train
num_epochs = 30

TrainNumberGain = 3

#From how many images one should belong to the validation set
EachNthVal = 10000

# numbper of processes extracting data from dataloader at the same time
num_workers = 5

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.6)
# alternative parameters 3 0.6
# alternative parameters 3 0.1
# alternative parameters 4 0.7

data_root = 'kaggle/working/plates/'

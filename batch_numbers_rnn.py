import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms
from torch.autograd import Variable

import data_prepare

class LSTM(nn.Module):
    def __init__(self, features, hidden_size, batch_size, num_layers, out_size):
        super(LSTM, self).__init__()
        """ torch.nn.LSTM(input_size, hidden_size, num_layers)
        # input_size - the number of input features per time-step
        # hidden_size - the number of LSTM blocks per layer
        # num_layers - the number of hidden layers
        """
        self.xh = torch.nn.LSTM(features, hidden_size, num_layers)
        self.hy = torch.nn.Linear(hidden_size, out_size)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers

    def __call__(self, xs, hidden):
        h, hidden = self.xh(xs, hidden)
        # LSTM output parameter has a shape of (seq_len, batch, hidden_size*num_directions)
        # Last element of the sequence is prediction
        y = self.hy(h[-1])
        return y, hidden
    
    def reset(self):
        self.hidden = (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)), Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))

def data_read():
    data_path = os.path.join("T-data.xls")
    # Download data for update
    # data_prepare.download_data(data_path)
    numbers_array = data_prepare.xls2array()
    numbers_array = numbers_array[:, :4]
    return numbers_array

def result2numbers(result):
    numbers = []
    for i in range(4):
        i = i * 10
        digit = result[-1, i:i+10]
        numbers.append(digit.argmax())

    return numbers
 

if __name__ == "__main__":
    # HYPER PARAMETERS
    EPOCH_NUM = 1000
    HIDDEN_SIZE = 1
    NUM_LAYERS = 1
    TIME_STEP = 3
    FEATURES = 40

    numbers_array = data_read()
    print(numbers_array[-1])
    BATCH_SIZE = len(numbers_array) - TIME_STEP

    onehotvectors = []
    for i in numbers_array:
        onehotvector = data_prepare.numbers2onehotvector(i)
        onehotvectors.append(onehotvector)

    train_x, train_t = [], []
    for i in range(BATCH_SIZE):
        train_x.append(onehotvectors[i:i+TIME_STEP])
        train_t.append(onehotvectors[i+TIME_STEP])
    train_x = np.array(train_x, dtype="float32")
    random_batch_index = np.array(range(len(train_x)))

    # RNN settings
    model = LSTM(features=FEATURES, hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, num_layers=NUM_LAYERS, out_size=40)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    st = datetime.datetime.now()

    for epoch in range(EPOCH_NUM):
        x, t = [], []
        
        # Random batchfy
        random_batch_index = np.random.permutation(random_batch_index)
        for i in random_batch_index:
            x.append(train_x[i])
            t.append(train_t[i])
        
        x = np.array(x, dtype="float32")
        t = np.array(t, dtype="float32")

        # Reshape (TIME_STEP, BATCH_SIZE) -> (TIME_STEP, BATCH_SIZE, FEATURES)
        x = x.transpose(1, 0, 2)
        # Reshape (BATCH_SIZE) -> (OUTPUT_SIZE, BATCH_SIZE)
        t = t.transpose(1, 0)

        # Convert 2 Pytorch Variable
        x = Variable(torch.from_numpy(x))
        t = Variable(torch.from_numpy(t))

        total_loss = 0
        #model.reset()
        y, output_hidden = model(x, hidden=None)
        loss = criterion(y, t)
        loss.backward()
        total_loss = total_loss + loss.data.numpy()[0]
        optimizer.step()
        if (epoch+1) % 100 == 0:
            ed = datetime.datetime.now()
            print("epoch:\t{}\ttotal_loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
            st = datetime.datetime.now()

    y = y.data.numpy()
    numbers = result2numbers(y)
    # y = np.round(y)
   # numbers = data_prepare.onehotvector2numbers(y[-1])
    #print(y[-1])
    print(numbers)

    # Prediction
    # latest = []
    # for i in range(3):
        # i = i - 3
        # latest.append(onehotvectors[i])
    # latest_batch = []
    # latest_batch.append(latest)
    # latest_batch = np.array(latest_batch, dtype="float32")
    # print(latest_batch)
    # latest_batch = latest_batch.transpose(1, 0, 2)
    # latest_batch = Variable(torch.from_numpy(latest_batch))
    # prediction = model(latest_batch, output_hidden[0])

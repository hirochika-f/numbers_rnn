import os
import datetime
import tqdm
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

    def forward(self, xs, hidden):
        # h, out_hidden = self.xh(xs, self.hidden)
        h, out_hidden = self.xh(xs, hidden)
        # LSTM output parameter has a shape of (seq_len, batch, hidden_size*num_directions)
        # Last element of the sequence is prediction
        y = self.hy(h[-1])
        return y, out_hidden
    
    def reset(self):
        self.hidden = (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)), 
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))

def data_read():
    data_path = os.path.join("T-data.xls")
    # Download data for update
    # data_prepare.download_data(data_path)
    numbers_array = data_prepare.xls2array()
    numbers_array = numbers_array[:, :4]
    return numbers_array

def result2numbers(result):
    """ RNN output 2 decimal. Only latest data. """
    numbers = []
    for i in range(4):
        i = i * 10
        digit = result[-1, i:i+10]
        numbers.append(digit.argmax())

    return numbers
 

if __name__ == "__main__":
    # HYPER PARAMETERS
    EPOCH_NUM = 1
    HIDDEN_SIZE = 5
    NUM_LAYERS = 1
    TIME_STEP = 3
    FEATURES = 40
    BATCH_SIZE = 3

    # Load NUMBERS
    numbers_array = data_read()
    print(numbers_array[-1])

    # NUMBERS to one hot vector
    onehotvectors = []
    for i in numbers_array:
        onehotvector = data_prepare.numbers2onehotvector(i)
        onehotvectors.append(onehotvector)

    # RNN settings
    model = LSTM(features=FEATURES, 
            hidden_size=HIDDEN_SIZE, 
            batch_size=BATCH_SIZE, 
            num_layers=NUM_LAYERS, 
            out_size=FEATURES)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Number of TIME_STEP data group
    DATA_LENGTH = len(numbers_array) - TIME_STEP

    # Create groups that have TIME_STEP data
    train_x, train_t = [], []
    for i in range(DATA_LENGTH):
        train_x.append(onehotvectors[i:i+TIME_STEP])
        train_t.append(onehotvectors[i+TIME_STEP])
    train_x = np.array(train_x, dtype="float32")

    # train_x indicies for randomly batchfy
    random_batch_indicies = np.array(range(len(train_x)-(BATCH_SIZE-1)))

    # Standard time to print times for each iteration
    st = datetime.datetime.now()

    for epoch in range(EPOCH_NUM):
        total_loss = 0

        # Randomized indicies for batchfy
        random_batch_indicies = np.random.permutation(random_batch_indicies)

        # Batch learning
        for i in random_batch_indicies:
            x, t = [], []
            for j in range(i, i+BATCH_SIZE):
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

            # model.reset()
            model.zero_grad()
            y, output_hidden = model(x, hidden=None)
            loss = criterion(y, t)
            loss.backward()
            total_loss = total_loss + loss.data.numpy()[0]
            optimizer.step()

        # Print loss each 100 epoch
        if (epoch+1) % 1 == 0:
            ed = datetime.datetime.now()
            print("epoch:\t{}\ttotal_loss:\t{}\ttime:\t{}".format(epoch+1, total_loss/len(random_batch_indicies), ed-st))
            st = datetime.datetime.now()

    # y = y.data.numpy()
    # numbers = result2numbers(y)
    # y = np.round(y)
    # numbers = data_prepare.onehotvector2numbers(y[-1])
    # print(y[-1])
    # print(numbers)

    # Construct data to prediction
    latest_data = []
    latest_data_num = (TIME_STEP + BATCH_SIZE - 1) * -1
    latest_onehotvectors = onehotvectors[latest_data_num:]
    for k in range(len(latest_onehotvectors)-TIME_STEP+1):
        latest_data.append(onehotvectors[k:k+TIME_STEP])
    latest_data = np.array(latest_data, dtype="float32")
    latest_data = latest_data.transpose(1, 0, 2)
    latest_data = Variable(torch.from_numpy(latest_data))

    # Prediction
    prediction, predicted_hidden = model(latest_data, output_hidden)
    prediction = prediction.data.numpy()
    predicted_numbers = result2numbers(prediction)
    print(predicted_numbers)

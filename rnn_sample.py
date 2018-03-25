import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, features, hidden_size, out_size):
        super(LSTM, self).__init__()
        """ torch.nn.LSTM(input_size, hidden_size, num_layers)
        # input_size - the number of input features per time-step
        # hidden_size - the number of LSTM blocks per layer
        # num_layers - the number of hidden layers
        """
        self.xh = torch.nn.LSTM(features, hidden_size, 1)
        self.hy = torch.nn.Linear(hidden_size, out_size)
        self.hidden_size = hidden_size

    def __call__(self, xs):
        h, self.hidden = self.xh(xs, self.hidden)
        # LSTM output parameter has a shape of (seq_len, batch, hidden_size*num_directions)
        # Last element of the sequence is prediction
        y = self.hy(h[-1])
        return y

    def reset(self):
        self.hidden = (Variable(torch.zeros(1, 490, self.hidden_size)), Variable(torch.zeros(1, 490, self.hidden_size)))

# HYPER PARAMETERS
EPOCH_NUM = 1000
HIDDEN_SIZE = 1
BATCH_SIZE = 490
TIME_STEP = 10
FEATURES = 1

# Data preparation
train_data = np.array([np.sin(i*2*np.pi/50) for i in range(50)]*10)
train_x, train_t = [], []
# Divide data into TIME_STEPs and prepare label next to the last
for i in range(0, len(train_data)-TIME_STEP):
    train_x.append(train_data[i:i+TIME_STEP])
    train_t.append(train_data[i+TIME_STEP])
train_x = np.array(train_x, dtype="float32")
train_t = np.array(train_t, dtype="float32")
# Transpose to input shape (TIME_STEP, BATCH_SIZE)
train_x = train_x.transpose()
N = len(train_x)

# RNN settings
model = LSTM(features=FEATURES, hidden_size=HIDDEN_SIZE, out_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

st = datetime.datetime.now()

for epoch in range(EPOCH_NUM):
    x, t = [], []

    # Batchfy
    for i in range(TIME_STEP):
        index = np.random.randint(0, N)
        x.append(train_x[index])
    x = np.array(x, dtype="float32")
    print(x.shape)
    # Reshape (TIME_STEP, BATCH_SIZE) -> (TIME_STEP, BATCH_SIZE, FEATURES)
    x = np.reshape(x, (TIME_STEP, BATCH_SIZE, 1))
    # Reshape (BATCH_SIZE) -> (OUTPUT_SIZE, BATCH_SIZE)
    t = np.reshape(train_t, (1, 490))
    x = Variable(torch.from_numpy(x))
    t = Variable(torch.from_numpy(t))
    total_loss = 0
    model.reset()
    y = model(x)
    loss = criterion(y, t)
    loss.backward()
    total_loss = total_loss + loss.data.numpy()[0]
    optimizer.step()
    if (epoch+1) % 100 == 0:
        ed = datetime.datetime.now()
        print("epoch:\t{}\ttotal_loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
        st = datetime.datetime.now()

# Plot predictions
y = y.data.numpy()
print(y)
plt.plot(y)
plt.show()

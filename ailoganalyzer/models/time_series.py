import pickle
from adtk.data import validate_series
from adtk.detector import SeasonalAD, LevelShiftAD, PersistAD, VolatilityShiftAD, AutoregressionAD
from pylab import show
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

def preprocess_TS(training_set, seq_length, sc = None):
    if sc == None:
        sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)



    seq_length = 4
    x, y = sliding_windows(training_data, seq_length)

    train_size = int(len(y) * 0.67)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))
    return (dataX,dataY), sc

def train_TS_LSTM(path, dataX ):
    num_epochs = 2000
    learning_rate = 0.01

    input_size = 1
    hidden_size = 2
    num_layers = 1

    num_classes = 1
    trainX, trainY = dataX


    lstm = timeSerie(num_classes, input_size, hidden_size, num_layers)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(outputs, trainY)

        loss.backward()

        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    print("save model")
    torch.save(lstm.state_dict(), path)

def test_TS_LSTM(path, data, sc):
    num_epochs = 2000
    learning_rate = 0.01

    input_size = 1
    hidden_size = 2
    num_layers = 1

    num_classes = 1
    dataX, dataY = data

    lstm = timeSerie(num_classes, input_size, hidden_size, num_layers)
    lstm.load_state_dict(torch.load(path))

    lstm.eval()
    train_predict = lstm(dataX)

    data_predict = train_predict.data.numpy()
    dataY_plot = dataY.data.numpy()

    data_predict = sc.inverse_transform(data_predict)
    dataY_plot = sc.inverse_transform(dataY_plot)

    plt.axvline(x=3000, c='r', linestyle='--')

    plt.plot(dataY_plot)
    plt.plot(data_predict)
    plt.suptitle('Time-Series Prediction')
    plt.show()


class timeSerie(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(timeSerie, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 365

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out



def save_time_serie(time_serie, filename):
    with open(filename, 'wb') as f:
        pickle.dump(time_serie, f)

def load_time_serie(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class TimeSerie():
    def __init__(self, model = "paternChange"):
        self.s_test = None
        self.anomalies = None
        self.s_train = None
        self.model_name = model

    def fit(self, s_train):
        self.s_train = s_train = validate_series(s_train)
        if self.model_name == 'seasonal':
            # changer si pas seasonal
            self.model = SeasonalAD()
            try:
                self.model.fit(s_train)
            except RuntimeError:
                self.model_name = "paternChange"

        if self.model_name == "paternChange":
            self.model = VolatilityShiftAD("1D")
        if self.model_name == "levelShift":
            self.model = LevelShiftAD(10)
        if self.model_name == "spike":
            self.model = PersistAD()
        if self.model_name == "cyclic":
            self.model = AutoregressionAD()
        self.model.fit(s_train)

    def anomaly(self, s_test):
        self.s_test = validate_series(s_test)
        self.anomalies = self.model.detect(self.s_test)

    def visualize(self, which = "train", anomaly = True):
        print(self.s_train)
        print(type(self.s_train))
        if which == "test":
            if anomaly:
                plot(self.s_test, anomaly=self.anomalies, anomaly_color="red", anomaly_tag="marker")
            else:
                if anomaly:
                    plot(self.s_test)
        else:
            plot(self.s_train)
        show()

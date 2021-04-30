import pickle
from pylab import where, append, sort
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
    x, y = sliding_windows(training_data, seq_length)

    dataX = Variable(torch.Tensor(np.array(x)).to(device))
    dataY = Variable(torch.Tensor(np.array(y)).to(device))
    return (dataX,dataY), sc

def train_TS_LSTM(path, dataX, options):
    num_epochs = 100
    learning_rate = 0.001

    input_size = 1
    hidden_size = 100
    num_layers = 1

    num_classes = 1
    trainX, trainY = dataX


    lstm = timeSerie(num_classes, input_size, hidden_size, num_layers).to(device)

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
        if epoch % 10 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    print("save model")
    torch.save(lstm.state_dict(), path)

def load_model_TS(path):

    input_size = 1
    hidden_size = 100
    num_layers = 1

    num_classes = 1

    lstm = timeSerie(num_classes, input_size, hidden_size, num_layers)
    lstm.load_state_dict(torch.load(path))
    lstm.to(device)
    return lstm

def test_TS_LSTM(path, data, sc, intervalle = []):

    dataX, dataY = data
    print(dataX)
    print(dataY)

    lstm = load_model_TS(path)
    lstm.eval()
    train_predict = lstm(dataX)

    data_predict = train_predict.cpu().data.numpy()
    dataY_plot = dataY.cpu().data.numpy()

    borne_inf, borne_sup = intervalle
    dif = data_predict - dataY_plot
    anomalyX = append(where(dif < borne_inf), where(dif > borne_sup))
    #print("anoma",list(anomalyX))

    data_predict = sc.inverse_transform(data_predict)
    dataY_plot = sc.inverse_transform(dataY_plot)
    anomalyY = data_predict[anomalyX]

    #plt.axvline(x=3000, c='r', linestyle='--')

    plt.plot(dataY_plot)
    plt.plot(data_predict)
    plt.scatter(anomalyX,anomalyY, color="red")
    plt.suptitle('Time-Series Prediction')
    plt.show()

def compute_normal_interval_TS(path, data):
    dataX, dataY = data
    lstm = load_model_TS(path)
    lstm.eval()
    prediction = lstm(dataX)
    data_predict = prediction.cpu().data.numpy()
    data_true = dataY.cpu().data.numpy()
    error_matrix = data_predict - data_true
    error_matrix = sort(error_matrix.reshape(1,-1)[0])
    print(error_matrix)
    sep = int(len(error_matrix) * 0.0)
    return error_matrix[sep], error_matrix[len(error_matrix)-sep-1]

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
            self.num_layers, x.size(0), self.hidden_size).to(device))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))

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

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import lightning as L


class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights

class DeepLogNet(nn.Module):
    def __init__(self, num_keys, hidden_size=64, num_layers=2):
        super(DeepLogNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(1,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, sequential):
        input0 = sequential
        out, _ = self.lstm(input0)
        out = self.fc(out[:, -1, :])
        return out


class LogAnomalyNet(nn.Module):
    def __init__(self, num_keys, hidden_size=128, num_layers=2):
        super(LogAnomalyNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm0 = nn.LSTM(1,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.lstm1 = nn.LSTM(300,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, num_keys)
        self.attention_size = self.hidden_size

        self.w_omega = Variable(
            torch.zeros(self.hidden_size, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

    def forward(self, semantic, quantitative):
        input0, input1 = quantitative, semantic

        out0, _ = self.lstm0(input0)

        out1, _ = self.lstm1(input1)
        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out


# define the LightningModule
class LogSequence(L.LightningModule):
    def __init__(self, net, optimizer_fun="adam", lr=0.001):
        super().__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        if optimizer_fun not in ["sgd", "adam"]:
            raise ValueError("optimizer must be one of sgd or adam")
        self.optimizer_fun = optimizer_fun
        self.lr = lr
        
        
    def forward(self, x):
        return self.net(**x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log("train_loss", round(loss.item(),4), prog_bar=True, on_step=False, on_epoch=True)
        # Logging to TensorBoard (if installed) by default
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        pred = self.is_anomaly_batch(y, output)
        acc = sum(pred)/len(pred)
        self.log_dict({'test_loss': loss, 'test_acc': acc}, on_epoch=True, on_step=False)
    
    def is_anomaly(self, data, mode="candidate", num_candidates = 10, proba_th = 0.1):
        x,y = data
        y = y["sequential"]
        x={a:b.unsqueeze(0) for a, b in x.items()}

    
    def is_anomaly_batch(self, labels, output, mode="candidate", num_candidates = 10, proba_th = 0.1):
        output = torch.softmax(output, dim=1)
        if mode == "candidate":
            output = torch.argsort(output, descending=True)[:,:num_candidates]
            x= torch.tensor([label in elt for label, elt in zip(labels, output)])
            #print(x)
            return x
        elif mode == "proba":
            return torch.tensor([elt[label] >= proba_th for elt,label in zip(labels,output)])
        else:
            raise ValueError("mode parameter must be one of 'candidate' or 'proba'.")

    def configure_optimizers(self):
        if self.optimizer_fun == 'sgd':
            optimizer = torch.optim.SGD(self.net.parameters(),
                                             lr=self.lr,
                                             momentum=0.5)
        elif self.optimizer_fun == 'adam':
            optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError
        return optimizer


class DeepLog(LogSequence):
    def __init__(self, num_keys, prefix_file=None, num_candidates=9, window_size=10, **learning_params):
        net = DeepLogNet(num_keys)
        super().__init__(net, **learning_params)

    def forward(self, data):
        return self.net(data["sequential"])


class LogAnomaly(LogSequence):
    def __init__(self, num_keys, prefix_file=None, num_candidates=9, window_size=10, **learning_params):
        net = LogAnomalyNet(num_keys)
        super().__init__(net, **learning_params)
    
    def forward(self, data):
        return self.net(data["semantic"], data["quantitative"])
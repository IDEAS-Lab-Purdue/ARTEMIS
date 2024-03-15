# MLP
import torch
import torch.nn as nn

class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1541, 2064, dtype=torch.float64)
        self.hidden2 = nn.Linear(2064, 1032, dtype=torch.float64)
        self.hidden3 = nn.Linear(1032, 516, dtype=torch.float64)
        self.hidden4 = nn.Linear(516, 258, dtype=torch.float64)
        self.hidden5 = nn.Linear(258, 129, dtype=torch.float64)
        self.output = nn.Linear(129, 5, dtype=torch.float64)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.output(x)
        return x

class Artemis(nn.Module):
    def __init__(self):
        super().__init__()
        # self.hidden1 = nn.Linear(1541, 2064, dtype=torch.float64)
        # self.hidden2 = nn.Linear(2064, 1032, dtype=torch.float64)
        # self.hidden3 = nn.Linear(1032, 516, dtype=torch.float64)
        # self.hidden4 = nn.Linear(516, 258, dtype=torch.float64)
        # self.hidden5 = nn.Linear(258, 129, dtype=torch.float64)
        # self.output = nn.Linear(129, 5, dtype=torch.float64)
        
        self.td1 = nn.Linear(1536, 524)
        
        #self.td2 = nn.Linear(2064, 1062)
        #self.td3 = nn.Linear(1062, 256)
        self.td4 = nn.Linear(524, 250)
        #self.td5 = nn.Linear(64, 56)
        self.normalize = nn.InstanceNorm1d(2)
        self.hidden1 = nn.Linear(255, 8)
        self.hidden2 = nn.Linear(8, 5)
        
        self.output = nn.Softmax()

    def forward(self, text_x, vitals_x):
        #text_x = (text_x - text_x.mean())/(text_x.std())
        t1 = self.td1(text_x)
        t1 = self.normalize(t1)
        #t1 = self.td2(t1)
        #t1 = self.td3(t1)
        t1 = self.td4(t1)
        t1 = self.normalize(t1)
        #t1 = self.td5(t1)
        #t1 = self.normalize(t1)
        model_in = torch.cat((t1, vitals_x), dim=1)
        #print("MODEL IN = ", model_in.shape)
        #model_in = self.normalize(model_in)
        out = self.hidden1(model_in)
        #out = self.normalize(out)
        out = self.hidden2(out)
        out = self.output(out)
        
        return out
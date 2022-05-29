import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


def make_hidden_layers(num_hidden_layers=1, hidden_size=5, prefix="y"):
    block = nn.Sequential()
    for i in range(num_hidden_layers):
        block.add_module(prefix+"_"+str(i), nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.BatchNorm1d(hidden_size),nn.ReLU()))
    return block

class Conv1(nn.Module):
    def __init__(self, input_size=1, num_classes=20):
        super(Conv1, self).__init__()
        #self.avgpool = nn.AdaptiveAvgPool1d(16 * num_classes)
        num_classes = 32
        self.fc1 = nn.Conv1d(input_size, 2 * num_classes, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(2 * num_classes)
        self.ac = nn.ReLU()
        self.fc2 = nn.Conv1d(2 * num_classes, 4 * num_classes, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(4 * num_classes)
        self.fc3 = nn.Conv1d(4 * num_classes, 8 * num_classes, kernel_size=7, stride=2)
        self.bn3 = nn.BatchNorm1d(8 * num_classes)
        self.mp = nn.AdaptiveMaxPool1d(output_size=2)
        # self.fc1 = nn.Linear(50, 40)
        # self.bn1 = nn.BatchNorm1d(40)
        # self.ac = nn.Softsign()
        # self.fc2 = nn.Linear(40, 30)
        # self.bn2 = nn.BatchNorm1d(30)
        self.fc4 = nn.Linear(16 * num_classes, 2)


    def forward(self, x):
        out = x
        out = self.fc1(out)
        #print(out.shape)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        #print(out.shape)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.fc3(out)
        #print(out.shape)
        out = self.bn3(out)
        out = self.ac(out)
        out = self.mp(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc4(out)
        return out
        



class Uci(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        num_hidden_layers = 10
        hidden_size = 150
        self.recon_fc1 = nn.Linear(input_size, hidden_size)
        self.recon_h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="recon")
        self.recon_fc2 = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        out = x
        out = F.relu(self.recon_fc1(out))
        out = self.recon_h_layers(out)
        y = self.recon_fc2(out)
        return y
        
        
class U_Encoder(nn.Module):
    def __init__(self, a_dim=2, z_dim=2, u_dim=2, y_dim=2, w_dim=2, by_dim=2, num_hidden_layers=1, hidden_size=5):
        super().__init__()
        # input_size = a_dim + z_dim
        input_size = z_dim
        num_classes = 20
        self.fc1 = nn.Linear(input_size, 8 * num_classes)
        self.bn1 = nn.BatchNorm1d(8 * num_classes)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(8 * num_classes, 4 * num_classes)
        self.bn2 = nn.BatchNorm1d(4 * num_classes)
        self.fc_mu = nn.Linear(4 * num_classes, u_dim)  # fc21 for mean of U
        self.fc_logvar = nn.Linear(4 * num_classes, u_dim)  # fc22 for log variance of U
        
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.h_layers = make_hidden_layers(num_hidden_layers, hidden_size=hidden_size, prefix="u")
        # self.fc_mu = nn.Linear(hidden_size, u_dim)  # fc21 for mean of U
        # self.fc_logvar = nn.Linear(hidden_size, u_dim)  # fc22 for log variance of U

    def forward(self, az):
        out = az
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


class Y_Bicoder(nn.Module):
    def __init__(self, a_dim=2, z_dim=2, u_dim=2, y_dim=2, w_dim=2, by_dim=2, num_hidden_layers=1, hidden_size=5):
        super().__init__()
        input_size = z_dim + u_dim
        # input_size = z_dim
        num_classes = 20
        self.fc1 = nn.Linear(input_size, 8 * num_classes)
        self.bn1 = nn.BatchNorm1d(8 * num_classes)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(8 * num_classes, 4 * num_classes)
        self.bn2 = nn.BatchNorm1d(4 * num_classes)
        self.fc3 = nn.Linear(4 * num_classes, y_dim)

    def forward(self, uz):
        # out = torch.cat((u, z), dim=1
        out = uz
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        y = self.fc3(out)
        return y


class AZ_Decoder(nn.Module):
    def __init__(self, a_dim=2, z_dim=2, u_dim=2, y_dim=2, w_dim=2, by_dim=2, num_hidden_layers=1, hidden_size=5):
        super().__init__()
        input_size = u_dim
        num_classes = 20
        self.fc1 = nn.Linear(input_size, 8 * num_classes)
        self.bn1 = nn.BatchNorm1d(8 * num_classes)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(8 * num_classes, 4 * num_classes)
        self.bn2 = nn.BatchNorm1d(4 * num_classes)
        self.fc3 = nn.Linear(4 * num_classes, z_dim)

    def forward(self, u):
        out = u
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        az = self.fc3(out)
        return az

class uW_Decoder(nn.Module):
    def __init__(self, a_dim=2, z_dim=2, u_dim=2, y_dim=2, w_dim=2, by_dim=2, num_hidden_layers=1, hidden_size=5):
        super().__init__()
        input_size = u_dim + z_dim
        num_classes = 20
        self.fc1 = nn.Linear(input_size, 8 * num_classes)
        self.bn1 = nn.BatchNorm1d(8 * num_classes)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(8 * num_classes, 4 * num_classes)
        self.bn2 = nn.BatchNorm1d(4 * num_classes)
        self.fc3 = nn.Linear(4 * num_classes, w_dim)

    def forward(self, u, z):
        out = torch.cat((u, z), dim=1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        w = self.fc3(out)
        return w

class W_Decoder(nn.Module):
    def __init__(self, a_dim=2, z_dim=2, u_dim=2, y_dim=2, w_dim=2, by_dim=2, num_hidden_layers=1, hidden_size=5):
        super().__init__()
        input_size = a_dim + u_dim + z_dim + y_dim
        num_classes = 20
        self.fc1 = nn.Linear(input_size, 8 * num_classes)
        self.bn1 = nn.BatchNorm1d(8 * num_classes)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(8 * num_classes, 4 * num_classes)
        self.bn2 = nn.BatchNorm1d(4 * num_classes)
        self.fc3 = nn.Linear(4 * num_classes, w_dim)

    def forward(self, u, az, y):
        out = torch.cat((u, az, y), dim=1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        w = self.fc3(out)
        return w

class Bar_Y_Decoder(nn.Module):
    def __init__(self, a_dim=2, z_dim=2, u_dim=2, y_dim=2, w_dim=2, by_dim=2, num_hidden_layers=1, hidden_size=5):
        super().__init__()
        input_size = z_dim + y_dim + w_dim + a_dim
        num_classes = 20
        self.fc1 = nn.Linear(input_size, 8 * num_classes)
        self.bn1 = nn.BatchNorm1d(8 * num_classes)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(8 * num_classes, 4 * num_classes)
        self.bn2 = nn.BatchNorm1d(4 * num_classes)
        self.fc3 = nn.Linear(4 * num_classes, by_dim)

    def forward(self, z, y, w):
        out = torch.cat((z, y, w), dim=1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        by = self.fc3(out)
        return by

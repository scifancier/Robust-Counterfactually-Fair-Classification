import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from encoders import *
from torch.autograd import Variable
import numpy as np

class UciNet(nn.Module):
    def __init__(self, input_size=1):
        super(UciNet, self).__init__()
        hidden_dim = 20
        self.fc1 = nn.Linear(input_size, 8 * hidden_dim)
        self.bn1 = nn.BatchNorm1d(8 * hidden_dim)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(8 * hidden_dim, 4 * hidden_dim)
        self.bn2 = nn.BatchNorm1d(4 * hidden_dim)
        self.fc3 = nn.Linear(4 * hidden_dim, 2)


    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.fc3(out)
        return out


class yVAE(nn.Module):
    def __init__(self, a_dim=1, z_dim=2, u_dim=2, y_dim=2, w_dim=2, by_dim=2,
                 num_hidden_layers=1, hidden_size=5):
        super().__init__()
        self.u_encoder = U_Encoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.y_bicoder = Y_Bicoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.az_decoder = AZ_Decoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                     num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.w_decoder = W_Decoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.by_decoder = Bar_Y_Decoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                        num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)

    def _y_pred_reparameterize(self, y_pred):
        return F.gumbel_softmax(y_pred)

    def _u_reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def mixup_data(self, x, y, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''

        lam = np.random.beta(alpha, alpha)

        batch_size = x.size()[0]

        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def forward(self, az, z, w, y, mix_up=False):

        inputs, targets_a, targets_b, lam = self.mixup_data(z, y)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        y_pred = self.y_bicoder(inputs)

        w_pred = self.w_decoder(az, y_pred)

        by_pred = self.by_decoder(az, y_pred, w)

        return 0, y_pred, 0, w_pred, by_pred, 0, 0, targets_a, targets_b, lam

class uVAE(nn.Module):
    def __init__(self, a_dim=1, z_dim=2, u_dim=2, y_dim=2, w_dim=2, by_dim=2,
                 num_hidden_layers=1, hidden_size=5):
        super().__init__()
        self.u_encoder = U_Encoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.y_bicoder = Y_Bicoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.az_decoder = AZ_Decoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                     num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.w_decoder = W_Decoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.by_decoder = Bar_Y_Decoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                        num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)

        self.uw_decoder = uW_Decoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)

    def _y_pred_reparameterize(self, y_pred):
        return F.gumbel_softmax(y_pred)

    def _u_reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def mixup_data(self, x, y, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''

        lam = np.random.beta(alpha, alpha)

        batch_size = x.size()[0]

        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def forward(self, az, z, w, y, mix_up=False):

        mu, logvar = self.u_encoder(z)
        u = self._u_reparameterize(mu, logvar)
       
        uz = torch.cat((u, z), dim=1)
        y_pred = self.y_bicoder(uz)

        z_pred = self.az_decoder(u)

        w_pred = self.w_decoder(u, z, y_pred)

        return u, y_pred, z_pred, w_pred, 0, mu, logvar

class BaseVAE(nn.Module):
    def __init__(self, a_dim=1, z_dim=2, u_dim=2, y_dim=2, w_dim=2, by_dim=2,
                                   num_hidden_layers=1, hidden_size=5):
        super().__init__()
        self.u_encoder = U_Encoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.y_bicoder = Y_Bicoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.az_decoder = AZ_Decoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.w_decoder = W_Decoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)
        self.by_decoder = Bar_Y_Decoder(a_dim=a_dim, z_dim=z_dim, u_dim=u_dim, y_dim=y_dim, w_dim=w_dim, by_dim=by_dim,
                                   num_hidden_layers=num_hidden_layers, hidden_size=hidden_size)


    def _y_pred_reparameterize(self, y_pred):
        return F.gumbel_softmax(y_pred)

    def _u_reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std
    
    def mixup_data(self, x, y, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''

        lam = np.random.beta(alpha, alpha)


        batch_size = x.size()[0]
        
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

    def forward(self, az, z, w, y, mix_up=False):

        mu, logvar = self.u_encoder(z)
        u = self._u_reparameterize(mu, logvar)
        
        uz = torch.cat((u, z), dim=1)
        
        if mix_up:
            inputs, targets_a, targets_b, lam = self.mixup_data(uz, y)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            y_pred = self.y_bicoder(inputs)
            
            z_pred = self.az_decoder(u)
            
            w_pred = self.w_decoder(u, az, y_pred)

            by_pred = self.by_decoder(az, y_pred, w)

            return u, y_pred, z_pred, w_pred, by_pred, mu, logvar, targets_a, targets_b, lam
        else:
            y_pred = self.y_bicoder(uz)

            az_pred = self.az_decoder(u)

            w_pred = self.w_decoder(u, az, y_pred)

            by_pred = self.by_decoder(az, y_pred, w)

            return u, y_pred, az_pred, w_pred, by_pred, mu, logvar

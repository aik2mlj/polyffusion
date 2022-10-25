from torch import nn
from torch.distributions import Normal


class RnnEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(RnnEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.linear_var = nn.Linear(hidden_dim * 2, z_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

    def forward(self, x):
        x = self.gru(x)[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x).exp_()
        dist = Normal(mu, var)
        return dist

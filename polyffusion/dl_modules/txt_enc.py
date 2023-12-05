from torch import nn
from torch.distributions import Normal


class TextureEncoder(nn.Module):
    def __init__(self, emb_size, hidden_dim, z_dim, num_channel=10):
        """input must be piano_mat: (B, 32, 128)"""
        super(TextureEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, num_channel, kernel_size=(4, 12), stride=(4, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
        )
        self.fc1 = nn.Linear(num_channel * 29, 1000)
        self.fc2 = nn.Linear(1000, emb_size)
        self.gru = nn.GRU(emb_size, hidden_dim, batch_first=True, bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.linear_var = nn.Linear(hidden_dim * 2, z_dim)
        self.emb_size = emb_size
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

    def forward(self, pr):
        # pr: (bs, 32, 128)
        bs = pr.size(0)
        pr = pr.unsqueeze(1)
        pr = self.cnn(pr).view(bs, 8, -1)
        pr = self.fc2(self.fc1(pr))  # (bs, 8, emb_size)
        pr = self.gru(pr)[-1]
        pr = pr.transpose_(0, 1).contiguous()
        pr = pr.view(pr.size(0), -1)
        mu = self.linear_mu(pr)
        var = self.linear_var(pr).exp_()
        dist = Normal(mu, var)
        return dist

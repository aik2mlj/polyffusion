from torch import nn


class NaiveNN(nn.Module):
    def __init__(
        self,
        input_dim=512,
        output_dim=512,
    ):
        """Only two linear layers"""
        super(NaiveNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, z_x):
        output = self.linear1(z_x)
        output = self.linear2(output)
        # print(output_mu.size(), output_var.size())
        return output

import torch
from torch import nn
from torch.nn import functional as F
import math
from ipdb import set_trace

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        hidden = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(hidden, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

if __name__ == "__main__":
    model = Attention(128)
    inputs = torch.autograd.Variable(torch.arange(0, 1024).view(2, 4, 128))#.long()
    hidden = torch.autograd.Variable(torch.arange(0, 512).view(4, 128))#.long()
    outputs = model(hidden, inputs)
    from ipdb import set_trace
    set_trace()

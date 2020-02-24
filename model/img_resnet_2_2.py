import torch
import torchvision as tv
from torchvision import models as tvm
from torch import nn

from ipdb import set_trace

class ResNet34(nn.Module):  
    def __init__(self):  
        super(ResNet34, self).__init__()  
        model = tvm.resnet34()  
        model = tvm.resnet34(pretrained=True)
        model.fc = nn.Linear(2048, 2)
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):  
        x = self.resnet_layer(x)  
        return x  

if __name__ == "__main__":
    from config import opt
    model = ResNet34(opt)
    inputs = torch.autograd.Variable(torch.arange(0, 128*3*227*227).view(128, 3, 227, 227))
    outputs = model(inputs)
    print(outputs.size())
    

import torch
import torchvision as tv
from torchvision import models as tvm
from torch import nn
from model.img_resnet_2_2 import ResNet34
class ModelBuilder(nn.Module):  
    def __init__(self, text_model,img_model, opt):  
        super(ModelBuilder, self).__init__()  
        self.opt = opt
        self.text_model = text_model
        
        self.img_model = img_model
        
        self.fc1 = nn.Sequential(
            nn.Linear(1536,512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2816,512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096,512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        ) 
        self.fc4 = nn.Sequential(
            nn.Linear(2816,512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc5 = nn.Linear(512,2)
        
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,vid_input_ids, vid_segment_ids, vid_input_mask,cid_input_ids, cid_segment_ids, cid_input_mask,img1,img2):  
        _,b1 = self.text_model(vid_input_ids, vid_segment_ids, vid_input_mask, output_all_encoded_layers=False)
        i1 = self.img_model(img1)
        _,b2 = self.text_model(cid_input_ids, cid_segment_ids, cid_input_mask, output_all_encoded_layers=False)
        i2 = self.img_model(img2)
        b1_b2 = torch.cat((b1, b2), 1)
        b1_i2 = torch.cat((b1, i2.view(i2.size()[0], -1)), 1)
        i1_i2 = torch.cat((i1.view(i1.size()[0],-1), i2.view(i2.size()[0], -1)), 1)
        i1_b2 = torch.cat((i1.view(i1.size()[0],-1),b2), 1)
        
        fiture1 = self.fc1(b1_b2)
        fiture2 = self.fc2(b1_i2)
        fiture3 = self.fc3(i1_i2)
        fiture4 = self.fc4(i1_b2)
        
        out1 = self.fc5(fiture1)
        out2 = self.fc5(fiture2)
        out3 = self.fc5(fiture3)
        out4 = self.fc5(fiture4)
        fiture = torch.cat([fiture1.unsqueeze(2),fiture2.unsqueeze(2),fiture3.unsqueeze(2),fiture4.unsqueeze(2)], 2)
        fiture = fiture.max(dim = 2)[0]
        output = self.fc(fiture)
        return out1,out2,out3,out4,output

if __name__ == "__main__":
    from config import opt
    model = ModelBuilder(TextAttnRNNBN, ResNet34, opt)
    text = torch.autograd.Variable(torch.arange(0, 128*25).view(128, 25)).long()
    img = torch.autograd.Variable(torch.arange(0, 128*3*227*227).view(128, 3, 227, 227))
    outputs = model(text, img)
    print(outputs.size())
    

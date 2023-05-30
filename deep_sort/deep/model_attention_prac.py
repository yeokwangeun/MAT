from torchinfo import summary
import torch.nn as nn
import torch
import model as mt_

model = mt_.Net()
model_path = '/workspace/MAT/deep_sort/deep/checkpoint/ckpt.t7'
state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
model.load_state_dict(state_dict)

class New_Layer(nn.Module):
    def __init__(self,layer_, in_channel, out_channel):
        super(New_Layer,self).__init__()
        self.layer = layer_
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 4))
    def forward(self,x):
        x1 = self.layer(x)
        x2 = self.conv1x1(x1)
        # print(x2.size())
        x2 = self.avgpool(x2)
        return x1,x2

class Attention_model(nn.Module):
    def __init__(self,num_classes=134 ,reid=False):
        super(Attention_model,self).__init__()
        model = mt_.Net()
        model_path = '/workspace/MAT/deep_sort/deep/checkpoint/ckpt.t7'
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        model.load_state_dict(state_dict)
        self.conv = model.conv
        self.layer1 = New_Layer(model.layer1,64,512)
        self.layer2 = New_Layer(model.layer2,128,512)
        self.layer3 = New_Layer(model.layer3,256,512)
        self.layer4 = model.layer4
        self.avgpool = nn.AvgPool2d((8,4),1)
        self.classifier1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.classifier2 = model.classifier
        self.reid = reid
        # print(model)
    def forward(self, x):
        x = self.conv(x)
        x1,x1_ = self.layer1(x)
        x2,x2_ = self.layer2(x1)
        x3,x3_ = self.layer3(x2)
        x4 = self.layer4(x3)
        # print(x1.size(),x2.size(),x3.size(),x4.size())
        # print(x1_.size(),x2_.size(),x3_.size(),x4.size())
        x_ = torch.cat((x1_,x2_,x3_,x4),1)
        # print(x_.size())
        x_ = self.avgpool(x_)
        x_ = x_.view(x_.size(0),-1)
        # B x 128
        if self.reid:
            x_ = x_.div(x_.norm(p=2,dim=1,keepdim=True))
            return x_
        # 
        x_ = self.classifier1(x_)   #FC layer1     
        x_ = self.classifier2(x_)   #FC layer2 
        return x_
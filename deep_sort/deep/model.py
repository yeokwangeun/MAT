import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out,is_downsample=False):
        super(BasicBlock,self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out,c_out,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y),True)

def make_layers(c_in,c_out,repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i ==0:
            blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
        else:
            blocks += [BasicBlock(c_out,c_out),]
    return nn.Sequential(*blocks)

class Net(nn.Module):
    def __init__(self, num_classes=315 ,reid=False, low_fusion=None):
        super(Net,self).__init__()
        # 3 128 64
        self.low_fusion = low_fusion
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
        ) # -> 64 64 32
        get_out_layer = lambda in_channel: nn.Sequential(
            nn.Conv2d(in_channel, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 4))
        )
        self.layer1 = make_layers(64,64,2,False)
        self.out1 = get_out_layer(64)
        # -> 64 64 32
        self.layer2 = make_layers(64,128,2,False)
        self.out2 = get_out_layer(128)
        # -> 128 64 32
        self.layer3 = make_layers(128,256,2,False)
        self.out3 = get_out_layer(256)
        # -> 256 64 32
        self.layer4 = make_layers(256,512,2,False)
        # -> 512 64 32
        self.down1 = make_layers(512,512,2,True) 
        # -> 512 32 16
        self.down2 = make_layers(512,512,2,True) 
        # -> 512 16 8
        self.down3 = make_layers(512,512,2,True) 
        # -> 512 8 4
        self.avgpool = nn.AvgPool2d((8,4),1)
        # 512 1 1 
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
        self.concat_fusion_conv = nn.Conv2d(512*4, 512, kernel_size=1, stride=1, bias=False)
        self.gate_conv = nn.Conv3d(512, 1, kernel_size=(1, 1, 1))
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x1 = self.out1(x)
        x = self.layer2(x)
        x2 = self.out2(x)
        x = self.layer3(x)
        x3 = self.out3(x)
        x = self.layer4(x)
        x4 = x

        if self.low_fusion == "add":
            x = x1 + x2 + x3 + x4
        elif self.low_fusion == "concat":
            x = torch.cat([x1, x2, x3, x4], axis=1)
            x = self.concat_fusion_conv(x)
        elif self.low_fusion == "gate":
            stacked = torch.stack([x1, x2, x3, x4], axis=2)
            gate = self.gate_conv(stacked)
            gate = einops.repeat(gate.squeeze(1), "b n h w -> b d n h w", d=512)
            x = stacked * gate
            x = torch.sum(x, axis=2)
        else:
            pass
        
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x


class TripleNet(nn.Module):
    def __init__(self, reid=False, low_fusion=None):
        super(TripleNet,self).__init__()
        # 3 128 64
        self.low_fusion = low_fusion
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
        ) # -> 64 64 32
        get_out_layer = lambda in_channel: nn.Sequential(
            nn.Conv2d(in_channel, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 4))
        )
        self.layer1 = make_layers(64,64,2,False)
        self.out1 = get_out_layer(64)
        # -> 64 64 32
        self.layer2 = make_layers(64,128,2,False)
        self.out2 = get_out_layer(128)
        # -> 128 64 32
        self.layer3 = make_layers(128,256,2,False)
        self.out3 = get_out_layer(256)
        # -> 256 64 32
        self.layer4 = make_layers(256,512,2,False)
        # -> 512 64 32
        self.down1 = make_layers(512,512,2,True) 
        # -> 512 32 16
        self.down2 = make_layers(512,512,2,True) 
        # -> 512 16 8
        self.down3 = make_layers(512,512,2,True) 
        # -> 512 8 4
        self.avgpool = nn.AvgPool2d((8,4),1)
        # 512 1 1 
        self.reid = reid
        self.concat_fusion_conv = nn.Conv2d(512*4, 512, kernel_size=1, stride=1, bias=False)
        self.gate_conv = nn.Conv3d(512, 1, kernel_size=(1, 1, 1))
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x1 = self.out1(x)
        x = self.layer2(x)
        x2 = self.out2(x)
        x = self.layer3(x)
        x3 = self.out3(x)
        x = self.layer4(x)
        x4 = x

        if self.low_fusion == "add":
            x = x1 + x2 + x3 + x4
        elif self.low_fusion == "concat":
            x = torch.cat([x1, x2, x3, x4], axis=1)
            x = self.concat_fusion_conv(x)
        elif self.low_fusion == "gate":
            stacked = torch.stack([x1, x2, x3, x4], axis=2)
            gate = self.gate_conv(stacked)
            gate = einops.repeat(gate.squeeze(1), "b n h w -> b d n h w", d=512)
            x = stacked * gate
            x = torch.sum(x, axis=2)
        else:
            pass
        
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        if self.reid:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x
        return x


if __name__ == '__main__':
    net = Net()
    x = torch.randn(4,3,128,64)
    y = net(x)
    import ipdb; ipdb.set_trace()



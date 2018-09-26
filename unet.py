import torch
from torch import nn

class DoubleConv(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        return self.layers(x)
    
class Down(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        super(Down,self).__init__()
        self.conv = DoubleConv(in_channels,out_channels)
        self.down = nn.MaxPool2d(2)
    
    def forward(self,x):
        x1 = self.conv(x)
        x2 = self.down(x1)
        return x1,x2
        
        
class Up(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        super(Up,self).__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels,2,stride=2)
        self.conv =  DoubleConv(in_channels,out_channels)
        
    def forward(self,x1,x2):
        x1 = self.up(x1)
        #add relu
        
        #add some form of padding for odd tensors
        o = torch.cat([x2, x1], dim=1)
        o = self.conv(o)
        return o
        
        
class Unet(nn.Module):
    
    """ 
    Params:
        in_channels - for a RGB image this would be 3
        out_channels - should be the desired number of output classes.
    Foward Input:
        tensor - batch_size, in_channels, height, width 
        width and height need to be even numbers.
    Output:
        tensor - batch_size, out_channels, height, width
    """ 
    
    def __init__(self, in_channels=1,out_channels=2):
        super(Unet,self).__init__()
        
        #down
        self.d1 = Down(in_channels,64)
        self.d2 = Down(64,128)
        self.d3 = Down(128,256)
        self.d4 = Down(256,512)
        #across
        self.a1 = DoubleConv(512,1024)
        #up!
        self.u1 = Up(1024,512)
        self.u2 = Up(512,256)
        self.u3 = Up(256,128)
        self.u4 = Up(128,64)
        self.out = nn.Conv2d(64,out_channels,3,padding=1)
    
    def forward(self,x):
        x1,o = self.d1(x)
        x2,o = self.d2(o)
        x3,o = self.d3(o)
        x4,o = self.d4(o)
        o = self.a1(o)
        o = self.u1(o,x4)
        o = self.u2(o,x3)
        o = self.u3(o,x2)
        o = self.u4(o,x1)
        o = self.out(o)
        return o
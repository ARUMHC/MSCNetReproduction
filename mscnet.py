import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class FirstHalfEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Decoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, padding=2)
    self.conv1 = Conv(in_channels, in_channels)
    self.conv2 = Conv(2*in_channels, in_channels)


  def forward(self, x1, x2):

    x1 = self.conv1(x1)

    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)

    x = self.conv2(x)
    return self.up(x)

class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.outlayer = nn.Sequential( 
        nn.Conv2d(in_channels, out_channels, kernel_size=(2,1), padding=(4,3)),
        nn.Sigmoid())

  def forward(self, x):
    return self.outlayer(x)


class SpatialAttention(nn.Module):
  def __init__(self, n_channels, x_size):
    super().__init__()
    n_channels *=2
    self.r1 = nn.Sequential(
        nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False, dilation=1),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
        nn.Upsample(x_size, mode="bilinear")
    )
    self.r3 = nn.Sequential(
        nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False, dilation=3),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
        nn.Upsample(x_size, mode="bilinear")
    )
    self.r5 = nn.Sequential(
        nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False, dilation=5),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
        nn.Upsample(x_size, mode="bilinear")
    )
    self.r7 = nn.Sequential(
        nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False, dilation=7),
        nn.BatchNorm2d(n_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
        nn.Upsample(x_size, mode="bilinear")
    )

  def forward(self, x):

    x1 = self.r1(x)
    x3 = self.r3(x)
    x5 = self.r5(x)
    x7 = self.r7(x)

    x_out = torch.add(torch.add(torch.add(x1, x3), x5), x7)

    return torch.mul(x, x_out)

class ChannelAttetnion(nn.Module):

  def __init__(self, n_channels):
    super().__init__()

    self.blocks = nn.Sequential(
        nn.AvgPool2d(1),
        nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
        nn.Sigmoid()
    )
  def forward(self, x):
    x_out = self.blocks(x)
    return torch.mul(x, x_out)

class FusionBlock(nn.Module):
  def __init__(self, n_channels, x_size):
    super().__init__()

    self.channelatt1 = ChannelAttetnion(n_channels)
    self.channelatt2 = ChannelAttetnion(n_channels)
    self.spatial = SpatialAttention(n_channels, x_size)
    self.outlayers = Conv(2*n_channels, n_channels)

  def forward(self, x1, x2):

    x1 = self.channelatt1(x1)
    x2 = self.channelatt2(x2)
    

    x = torch.cat([x1, x2], dim=1)
    x = self.spatial(x)
    x = self.outlayers(x)

    return x

class MSCNet(nn.Module):
  def __init__(self, n_channels, n_classes = 1, inner_sizes=(38, 72, 150, 306), output_size=624):
    super().__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes

    ## unets
    # unet1
    self.beginning_conv_top = nn.Conv2d(self.n_channels, 64//2, kernel_size=3)
    
    # block 1 down
    self.enc1a_top = FirstHalfEncoder(64//2,128//2)
    self.enc1b_top = Conv(128//2,128//2)

    # block 2 down
    self.enc2a_top = FirstHalfEncoder(128//2,256//2)
    self.enc2b_top = Conv(256//2,256//2)

    # block 3 down
    self.enc3a_top = FirstHalfEncoder(256//2,512//2)
    self.enc3b_top = Conv(512//2,512//2)

    # block 4 down
    self.enc4a_top = FirstHalfEncoder(512//2,1024//2)
    self.enc4b_top = Conv(1024//2,1024//2)

    # block 4 up
    self.dec4_top = Decoder(1024//2, 512//2)
    self.dec3_top = Decoder(512//2, 256//2)
    self.dec2_top = Decoder(256//2, 128//2)
    self.dec1_top = Decoder(128//2, 64//2)

    # output

    self.out_top = nn.Sequential(OutConv(64//2, n_classes),
                                 torch.nn.Upsample((output_size, output_size)))

    # unet2
    
    self.beginning_conv_bot = nn.Conv2d(self.n_channels, 64//2, kernel_size=3)
    
    # block 1 down
    self.enc1a_bot = FirstHalfEncoder(64//2,128//2)
    self.enc1b_bot = Conv(128//2,128//2)

    # block 2 down
    self.enc2a_bot = FirstHalfEncoder(128//2,256//2)
    self.enc2b_bot = Conv(256//2,256//2)

    # block 3 down
    self.enc3a_bot = FirstHalfEncoder(256//2,512//2)
    self.enc3b_bot = Conv(512//2,512//2)

    # block 4 down
    self.enc4a_bot = FirstHalfEncoder(512//2,1024//2)
    self.enc4b_bot = Conv(1024//2,1024//2)

    # block 4 up
    self.dec4_bot = Decoder(1024//2, 512//2)
    self.dec3_bot = Decoder(512//2, 256//2)
    self.dec2_bot = Decoder(256//2, 128//2)
    self.dec1_bot = Decoder(128//2, 64//2)

    # output

    self.out_bot = nn.Sequential(OutConv(64//2, n_classes),
                                 torch.nn.Upsample((output_size, output_size)))

    ## fusion

    self.fusion4 = FusionBlock(1024//2, (inner_sizes[0], inner_sizes[0]))
    self.fusion3 = FusionBlock(512//2, (inner_sizes[1], inner_sizes[1]))
    self.fusion2 = FusionBlock(256//2, (inner_sizes[2], inner_sizes[2]))
    self.fusion1 = FusionBlock(128//2, (inner_sizes[3], inner_sizes[3]))

    self.sigmoid_top = nn.Sigmoid()
    self.sigmoid_bot = nn.Sigmoid()

  def forward(self, x):
    
    x_init_top = self.beginning_conv_top(x)
    x1a_top = self.enc1a_top(x_init_top)
    x1b_top = self.enc1b_top(x1a_top)

    x2a_top = self.enc2a_top(x1b_top)
    x2b_top = self.enc2b_top(x2a_top)

    x3a_top = self.enc3a_top(x2b_top)
    x3b_top = self.enc3b_top(x3a_top)

    x4a_top = self.enc4a_top(x3b_top)
    x4b_top = self.enc4b_top(x4a_top)

    x_init_bot = self.beginning_conv_bot(x)    
    x1a_bot = self.enc1a_bot(x_init_bot)
    x1b_bot = self.enc1b_bot(x1a_bot)

    x2a_bot = self.enc2a_bot(x1b_bot)
    x2b_bot = self.enc2b_bot(x2a_bot)

    x3a_bot = self.enc3a_bot(x2b_bot)
    x3b_bot = self.enc3b_bot(x3a_bot)

    x4a_bot = self.enc4a_bot(x3b_bot)
    x4b_bot = self.enc4b_bot(x4a_bot)

    fusion4 = self.fusion4(x4b_top, x4b_bot)

    x4_top = torch.add(x4b_top, fusion4)
    x4_bot = torch.add(x4b_bot, fusion4)

    x4_top = self.dec4_top(x4_top, x4a_top)
    x4_bot = self.dec4_bot(x4_bot, x4a_bot)

    fusion3 = self.fusion3(x4_top, x4_bot)

    x3_top = torch.add(x4_top, fusion3)        
    x3_bot = torch.add(x4_bot, fusion3)        

    x3_top = self.dec3_top(x3_top, x3a_top)
    x3_bot = self.dec3_bot(x3_bot, x3a_bot)

    fusion2 = self.fusion2(x3_top, x3_bot)

    x2_top = torch.add(x3_top, fusion2)
    x2_bot = torch.add(x3_bot, fusion2)

    x2_top = self.dec2_top(x2_top, x2a_top)
    x2_bot = self.dec2_bot(x2_bot, x2a_bot)

    fusion1 = self.fusion1(x2_top, x2_bot)

    x1_top = torch.add(x2_top, fusion1)
    x1_bot = torch.add(x2_bot, fusion1)

    x1_top = self.dec1_top(x1_top, x1a_top)
    x1_bot = self.dec1_bot(x1_bot, x1a_bot)

    x_top = self.out_top(x1_top)
    x_top = self.sigmoid_top(x_top)

    x_bot = self.out_top(x1_bot)
    x_bot = self.sigmoid_bot(x_bot)

    return x_top, x_bot

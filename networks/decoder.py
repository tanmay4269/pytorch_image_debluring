from torch import nn

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super(Block, self).__init__()

    self.conv_block1 = nn.Sequential(
      nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=0),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
    )

    self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride+1, padding=1, output_padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
      
  def forward(self, x, features):
    x = self.conv_block1(x)
    x = self.conv2(x)

    x += features
    x = self.bn2(x)
    x = self.relu(x)
    
    return x
    
class Decoder(nn.Module):
  def __init__(self): 
    super(Decoder, self).__init__()
    
    self.layer1 = Block(512, 256, stride=1)
    self.layer2 = Block(256, 128, stride=1)
    self.layer3 = Block(128, 64, stride=1)
    self.layer4 = Block(64, 64, stride=1)

    self.final_block = nn.Sequential(
      nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, output_padding=1),
      nn.BatchNorm2d(3),
      nn.ReLU()
    )

  def forward(self, encoder_features):
    features = encoder_features[::-1]

    x = self.layer1(features[0], features[1])
    x = self.layer2(x, features[2])
    x = self.layer3(x, features[3])
    x = self.layer4(x, features[4])
    
    x = self.final_block(x)
    
    return x 

from torch import nn
import torchvision.models as models

class ResnetEncoder(nn.Module):
  def __init__(self):
    super(ResnetEncoder, self).__init__()

    self.encoder = models.resnet18(weights="ResNet18_Weights.DEFAULT")

    for child in self.children():
      for param in child.parameters():
          param.requires_grad = False

  def forward(self, input_image):
    features = []
    
    x = self.encoder.conv1(input_image)
    x = self.encoder.bn1(x)
    features.append(self.encoder.relu(x))
    features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
    features.append(self.encoder.layer2(features[-1]))
    features.append(self.encoder.layer3(features[-1]))
    features.append(self.encoder.layer4(features[-1]))

    return features

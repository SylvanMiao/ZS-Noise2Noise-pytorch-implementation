import torch.nn as nn

class network(nn.Module):
  def __init__(self, in_channel, embed_channel = 48):
    super().__init__()
    self.act = nn.LeakyReLU(negative_slope= 0.2, inplace= True)
    self.conv1 = nn.Conv2d(in_channel, embed_channel, 3, padding=1)
    self.conv2 = nn.Conv2d(embed_channel, embed_channel, 3, padding=1)
    self.conv3 = nn.Conv2d(embed_channel, in_channel, 1)
  def forward(self, x):
    x = self.act(self.conv1(x))
    x = self.act(self.conv2(x))
    x = self.conv3(x)
    return x

